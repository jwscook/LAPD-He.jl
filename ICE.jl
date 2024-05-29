using Distributed, Dates
using Plots, Random, ImageFiltering, Statistics
using Dierckx, Contour, JLD2, DelimitedFiles

println("Starting at ", now())
# pitchanglecosine of 0 is all vperp, and (-)1 is (anti-)parallel to B field
const pitchanglecosine = try; parse(Float64, ARGS[1]); catch; cos(35π/180); end
@assert -1 <= pitchanglecosine <= 1
# thermal width of ring as a fraction of its speed # Dendy PRL 1993
const vthermalfractionz = try; parse(Float64, ARGS[2]); catch; 0.1; end
const vthermalfraction⊥ = try; parse(Float64, ARGS[3]); catch; 0.1; end
const name_extension = if length(ARGS) >= 4
  ARGS[4]
else
  "$(pitchanglecosine)_$(vthermalfractionz)_$(vthermalfraction⊥)"
end
const filecontents = [i for i in readlines(open(@__FILE__))]
const nprocsadded = div(Sys.CPU_THREADS, 2)

addprocs(nprocsadded, exeflags="--project")

@everywhere using ProgressMeter # for some reason must be up here on its own
@everywhere using StaticArrays
@everywhere using FastClosures
@everywhere using NLsolve
@everywhere begin
  using LinearMaxwellVlasov, LinearAlgebra, WindingNelderMead

  mₑ = LinearMaxwellVlasov.mₑ
  mi = 4*1836*mₑ

  n0 = 3e16
  B0 = 0.15
  ξ = 0.0233
  Z = 1 # singly charged Helium

  ni = n0 / (1.0 + Z * ξ)
  nf = ξ*ni
  @assert n0 ≈ Z*nf + ni
  Va = sqrt(B0^2/LinearMaxwellVlasov.μ₀/ni/mi)

  Ωe = cyclotronfrequency(B0, mₑ, -1)
  Ωi = cyclotronfrequency(B0, mi, 1)
  Ωf = cyclotronfrequency(B0, mi, Z)
  Πe = plasmafrequency(n0, mₑ, -1)
  Πi = plasmafrequency(ni, mi, 1)
  Πf = plasmafrequency(nf, mi, Z)
  vthe = thermalspeed(7, mₑ)
  vthi = thermalspeed(7, mi)
  vf = thermalspeed(15e3, mi)
  # pitchanglecosine = cos(pitchangle)
  # acos(pitchanglecosine) = pitchangle
  pitchanglecosine = Float64(@fetchfrom 1 pitchanglecosine)
  vf⊥ = vf * sqrt(1 - pitchanglecosine^2) # perp speed
  vfz = vf * pitchanglecosine # parallel speed
  vthermalfractionz = Float64(@fetchfrom 1 vthermalfractionz)
  vthermalfraction⊥ = Float64(@fetchfrom 1 vthermalfraction⊥)
  vfthz = vf * vthermalfractionz
  vfth⊥ = vf * vthermalfraction⊥

  electron_cold = ColdSpecies(Πe, Ωe)
  electron_warm = WarmSpecies(Πe, Ωe, vthe)
  electron_maxw = MaxwellianSpecies(Πe, Ωe, vthe, vthe)

  deuteron_cold = ColdSpecies(Πi, Ωi)
  deuteron_warm = WarmSpecies(Πi, Ωi, vthi)
  deuteron_maxw = MaxwellianSpecies(Πi, Ωi, vthi, vthi)

  fast_cold = ColdSpecies(Πf, Ωf)
  fast_maxw = MaxwellianSpecies(Πf, Ωf, vfthz, vfth⊥, vfz)
  fast_ringbeam = SeparableVelocitySpecies(Πf, Ωf,
    FBeam(vfthz, vfz),
    FRing(vfth⊥, vf⊥))
  fast_delta = SeparableVelocitySpecies(Πf, Ωi,
    FParallelDiracDelta(vfz),
    FPerpendicularDiracDelta(vf⊥))
  fast_rbdelta = SeparableVelocitySpecies(Πf, Ωi,
    FBeam(vfthz, vfz),
    FPerpendicularDiracDelta(vf⊥))

  Smmr = Plasma([electron_maxw, deuteron_maxw, fast_ringbeam])
  Smmd = Plasma([electron_maxw, deuteron_maxw, fast_delta])
  Smmh = Plasma([electron_maxw, deuteron_maxw, fast_rbdelta])

  w0 = abs(Ωi)
  k0 = w0 / abs(Va)

  gammamax = abs(Ωi) * 0.005
  gammamin = -gammamax / 4
  function bounds(ω0)
    lb = @SArray [ω0 - w0 / 2, gammamin]
    ub = @SArray [ω0 + w0 / 2, gammamax]
    return (lb, ub)
  end

  options = Options(memoiseparallel=false, memoiseperpendicular=true)

  function solve_given_kω(K, ωr, objective!)
    lb, ub = bounds(ωr)

    function boundify(f::T) where {T}
      @inbounds isinbounds(x) = all(i->0 <= x[i] <= 1, eachindex(x))
      maybeaddinf(x::U, addinf::Bool) where {U} = addinf ? x + U(Inf) : x
      bounded(x) = maybeaddinf(f(x), !isinbounds(x))
      return bounded
    end

    config = Configuration(K, options)

    ics = ((@SArray [ωr, gammamax*0.9]),
           (@SArray [ωr, gammamax*0.25]),
           (@SArray [ωr, gammamax*0.1]),
           (@SArray [ωr, -gammamax*0.1]))

    function unitobjective!(c, x::T) where {T}
      return objective!(c,
        T([x[i] * (ub[i] - lb[i]) + lb[i] for i in eachindex(x)]))
    end
    unitobjectivex! = x -> unitobjective!(config, x)
    boundedunitobjective! = boundify(unitobjectivex!)
    xtol_abs = (@SArray [1e-4, 1e-6])
    solutions = []

    @elapsed for ic ∈ ics
      @assert all(i->lb[i] <= ic[i] <= ub[i], eachindex(ic))
      neldermeadsol = WindingNelderMead.optimise(
        boundedunitobjective!, SArray((ic .- lb) ./ (ub .- lb)),
        1.1e-2 * (@SArray ones(2)); stopval=1e-15, timelimit=30,
        maxiters=200, ftol_rel=0, ftol_abs=0, xtol_rel=0, xtol_abs=xtol_abs)
      simplex, windingnumber, returncode, numiterations = neldermeadsol
      if (windingnumber == 1 && returncode == :XTOL_REACHED)# || returncode == :STOPVAL_REACHED
        c = deepcopy(config)
        minimiser = if windingnumber == 0
          WindingNelderMead.position(WindingNelderMead.bestvertex(simplex))
        else
          WindingNelderMead.centre(simplex)
        end
        unitobjective!(c, minimiser)
        push!(solutions, c)
      end
#      try
#        nlsolution = nlsolve(x->reim(boundedunitobjective!(x)),
#                             MArray((ic .- lb) ./ (ub .- lb)),
#                             xtol=1e-10)
#        if nlsolution.x_converged || nlsolution.f_converged
#          c = deepcopy(config)
#          objective!(c, nlsolution.zero .* (ub .- lb) .+ lb)
#          push!(solutions, c)
#        end
#      catch err
#        @info err
#      end
    end
    return solutions
  end

  function f2Dω!(config::Configuration, x::AbstractArray, plasma, cache)
    config.frequency = Complex(x[1], x[2])
    return det(tensor(plasma, config, cache))
  end

  function findsolutions(plasma)
    nk = 128 * 4
    nw = 64 * 2
    θ = 89.5 * π / 180
    ωrs = range(0.0, stop=50, length=nw) * w0
    ks = collect((1/2nk:1/nk:1-1/2nk) .* 2000 .+ 100) .* k0
    ks = vcat(collect(1/2nk:1/nk:1-1/2nk) .* 100 * k0, ks)
    ks = vcat(-ks, ks)
    kz⊥s = [abs(k) .* (sign(k) * cospi(θ / π), sinpi(θ / π)) for k in ks]

    # change order for better distributed scheduling
#    ks = shuffle(vcat([ks[i:nprocs():end] for i ∈ 1:nprocs()]...))
#    @assert length(ks) == nk
    solutions = @sync @showprogress @distributed (vcat) for ωr ∈ ωrs
      cache = Cache()
      objective! = @closure (C, x) -> f2Dω!(C, x, plasma, cache)
      innersolutions = Vector()
      for (ik, kz⊥) ∈ enumerate(kz⊥s)
        @assert kz⊥[2] >= 0
        #K = Wavenumber(wavenumber=k, propagationangle=θ)
        K = Wavenumber(parallel=kz⊥[1], perpendicular=kz⊥[2])
        output = solve_given_kω(K, ωr, objective!)
        isempty(output) && continue
        push!(innersolutions, output...)
      end
      innersolutions
    end
    return solutions
  end
end #@everywhere


Plots.gr()
function plotit(sols, file_extension=name_extension, fontsize=9)
  sols = sort(sols, by=s->imag(s.frequency))
  ωs = [sol.frequency for sol in sols]./w0
  kzs = [para(sol.wavenumber) for sol in sols]./k0
  k⊥s = [perp(sol.wavenumber) for sol in sols]./k0
  ks = [abs(sol.wavenumber) for sol in sols]./k0 .* sign.(kzs)

  kθs = atan.(k⊥s, kzs)

  msize = 1
  mshape = :square
  xlabel = "\$\\mathrm{Perpendicular\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  colorgrad = Plots.cgrad()

  xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"

  imaglolim = -1e-4
  wplotmax = ceil(Int, maximum(real, ωs) / 10) * 10
  kplotmax = ceil(Int, maximum(real, ks) / 50) * 50

  mask = shuffle(findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= wplotmax)))
  @warn "Scatter plots rendering with $(length(mask)) points."
  perm = sortperm(imag.(ωs[mask]))
  h0 = Plots.scatter(real.(ωs[mask][perm]), kzs[mask][perm],
    zcolor=imag.(ωs[mask][perm]), framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=0, markershape=:circle,
    c=colorgrad, xticks=(0:5:wplotmax), yticks=0:50:kplotmax,
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig("ICE2D_KFwplotmax_$file_extension.pdf")

  xlabel = "\$\\mathrm{Wavenumber} \\ [\\Omega_{i}/V_A]\$"
  ylabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  h0 = Plots.scatter(ks[mask][perm], real.(ωs[mask][perm]),
    zcolor=(imag.(ωs[mask][perm])), framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=0, markershape=:circle,
    c=colorgrad, xticks=-kplotmax:200:kplotmax, yticks=0:5:wplotmax,
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig("ICE2D_FKwplotmax_$file_extension.pdf")

  #xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  xlabel = "\$\\mathrm{Wavenumber} \\ [\\Omega_{i}/V_A]\$"
  ylabel = "\$\\mathrm{Growth\\ Rate} \\ [\\Omega_{i}]\$"
  mask = shuffle(findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= wplotmax)))
  h1 = Plots.scatter(real.(ks[mask]), imag.(ωs[mask]),
    zcolor=kzs[mask], framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=0, markershape=:circle,
    c=colorgrad, xticks=(0:5:wplotmax),
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig("ICE2D_Kwplotmax_$file_extension.pdf")

  colorgrad1 = Plots.cgrad([:cyan, :red, :blue, :orange, :green,
                            :black, :yellow])
  mask = shuffle(findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= wplotmax)))
  h2 = Plots.scatter(real.(ωs[mask]), imag.(ωs[mask]),
    zcolor=(real.(ωs[mask]) .- vfz/Va .* kzs[mask]), framestyle=:box, lims=:round,
    markersize=msize+1, markerstrokewidth=0, markershape=:circle,
    c=colorgrad1, clims=(0, 13), xticks=(0:5:wplotmax),
    xlabel=xlabel, ylabel=ylabel, legend=:topleft)
  Plots.plot!(legend=false)
  Plots.savefig("ICE2D_Fwplotmax_Doppler_$file_extension.pdf")

  xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  ylabel = "\$\\mathrm{Propagation\\ Angle} \\ [^{\\circ}]\$"

  colorgrad = Plots.cgrad([:cyan, :black, :darkred, :red, :orange, :yellow])
  mask = findall(@. (imag(ωs) > imaglolim) & (real(ωs) <= wplotmax))
  h4 = Plots.scatter(real.(ωs[mask]), kθs[mask] .* 180 / π,
    zcolor=imag.(ωs[mask]), lims=:round,
    markersize=msize, markerstrokewidth=0, markershape=mshape, framestyle=:box,
    c=Plots.cgrad([:black, :darkred, :red, :orange, :yellow]),
    clims=(0, maximum(imag.(ωs[mask]))),
    yticks=(0:10:180), xticks=(0:wplotmax), xlabel=xlabel, ylabel=ylabel)
  Plots.plot!(legend=false)
  Plots.savefig("ICE2D_TFwplotmax_$file_extension.pdf")

  function relative(p, rx, ry)
    xlims = Plots.xlims(p)
    ylims = Plots.ylims(p)
    return xlims[1] + rx * (xlims[2]-xlims[1]), ylims[1] + ry * (ylims[2] - ylims[1])
   end
  Plots.xlabel!(h1, "")
  Plots.xticks!(h1, 0:-1)
  if pitchanglecosine == -0.646
    xy_data = readdlm("data.csv", ',', Float64)
    x_data = xy_data[:, 1]
    y_data = xy_data[:, 2]
    y_data .-= minimum(y_data)
    y_data ./= (maximum(y_data) - minimum(y_data))
    y_data .*= maximum(imag.(ωs[mask]))
    x_data .*= 1e6 / (w0/2π)
    Plots.plot!(h1, x_data, y_data, color=:black)
  end
  Plots.annotate!(h1, [(relative(h1, 0.02, 0.95)..., text("(a)", fontsize, :black))])
  Plots.annotate!(h0, [(relative(h0, 0.02, 0.95)..., text("(b)", fontsize, :black))])
  Plots.plot(h1, h0, link=:x, layout=@layout [a; b])
  Plots.savefig("ICE2D_Combo_$file_extension.pdf")
end

if true
  @time plasmasols = findsolutions(Smmh)
  @show length(plasmasols)
  @time plotit(plasmasols)
  @save "solutions2D_$name_extension.jld" filecontents plasmasols w0 k0
  rmprocs(nprocsadded)
else
  rmprocs(nprocsadded)
  @load "solutions2D_$name_extension.jld" filecontents plasmasols w0 k0
  @time plotit(plasmasols)
end

println("Ending at ", now())


