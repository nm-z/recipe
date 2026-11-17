# build.jl: build.rs equivalent, two roles.
#  · Included by recipe.jl as module Recipe.BLD (sees SHARED: FL/FF/IF pack, MR, …):
#    templates(manifest::Dict, source::String) → Dict("amd:default" => IR, "amd:-f32" => …, "amd:gfx11-f16" => …,
#    "nvidia:default" => …) with the keys Rust put in RECIPE_AMD_IR / RECIPE_NV_IR. Parity: byte-identical to Rust's
#    OUT_DIR/recipe-{amd,nvidia}*.ll. No CPU target (compile_cpu and CPU_REPLACEMENTS are dropped).
#  · Run as `julia build.jl OUT`: packages the single executable OUT/recipe (PackageCompiler app + launcher.c payload).

# ---- BUILD section: port build.rs 151-982 (templates) here. Budget 52,000 chars for this whole file. ----
templates(m::AbstractDict, src::AbstractString) = Dict{String,String}()

# ---- packaging (Julia 1.13.4, PackageCompiler 2.4.5, cc, tar, zstd, libzstd) ----
pkg_id(u, n) = Base.require(Base.PkgId(Base.UUID(u), n))
pkg_env(Pkg, p, o) = (Pkg.activate(p); Pkg.resolve(); Pkg.activate(joinpath(o, "env")); Pkg.add(name="PackageCompiler", version="2.4.5"))
pkg_app(pc, p, a, j) = pc.create_app(p, a; executables=["recipe" => "julia_main"], precompile_execution_file=j, filter_stdlibs=true,
	include_lazy_artifacts=false, cpu_target=pc.default_app_cpu_target(), force=true)
function package_app(o)
	o = abspath(o); r = @__DIR__; p = joinpath(o, "pkg"); a = joinpath(o, "app"); z = joinpath(o, "payload.tar.zst"); j = joinpath(o, "precompile.jl")
	mkpath(p); for f in ("Project.toml", "recipe.jl", "build.jl", "amd-nv-cpu.ll") cp(joinpath(r, f), joinpath(p, f); force=true) end
	Base.invokelatest(pkg_env, pkg_id("44cfe95a-1eb2-52ea-b672-e2afdf69b78f", "Pkg"), p, o)
	write(j, "using Recipe\n")
	Base.invokelatest(pkg_app, pkg_id("9b87118b-4619-50d2-8e1e-99f35a4d4d9d", "PackageCompiler"), p, a, j)
	run(`cc -O2 -Wall -Wextra -Wno-format-truncation -o $o/launcher $r/launcher.c -lzstd`)
	run(pipeline(`tar --format=gnu --sort=name --owner=0 --group=0 --numeric-owner --mtime=@0 -C $a -cf - .`, `zstd -q -19 -T4 -f -o $z`))
	k = readchomp(`sha256sum $z`)[1:32]; run(`$o/launcher --pack $z $k recipe $o/recipe`)
	println("built ", joinpath(o, "recipe"))
end
abspath(PROGRAM_FILE) == @__FILE__() && package_app(isempty(ARGS) ? "out" : ARGS[1])
