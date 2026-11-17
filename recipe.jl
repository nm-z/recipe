# Julia port of nm-z/recipe (Rust recipe.rs + cli.rs). Tracking: nm-z/recipe#2.
# Branch limits: only Project.toml amd-nv-cpu.ll recipe.jl build.jl launcher.c, ≤1,000,000 chars (wc -m), stdlib deps only.
# Layout: SHARED (types, config, encodings, parity helpers, stubs), then one `@sec NAME begin … end` submodule per
# section, each under a "# ==== NAME ====" marker. Rules for section authors:
#  · Edit only your section; touch SHARED only when unavoidable and minimally. Keep within your char budget.
#  · A section sees every SHARED name (and earlier section modules) through `using`; a private definition that reuses
#    a SHARED name fails at load, so rename it. Never define a Base-exported name in SHARED (it would shadow Base).
#  · Implement a SHARED stub as `Recipe.name(args) = …` (any section); call it as plain `name(args)`.
#  · Reach another section's private name as `Recipe.SEC.name` (resolved at run time) and list such names in your header.
#  · Rust `Result`/`?` → throw: err(msg), require(cond, msg); `panic!`/`expect`/`unwrap` failures → panic(msg).
#  · Bits that reach output must match Rust: use rs (float→int `as`), libm l* (lexp, llog = ln, …), powi, ssum (Rust
#    `.sum()` order), fmax/fmin, fmtd/fmtq/fmte/fmtp (Rust `{}` `{:?}` `{:e}` `{:.N}`). Julia's exp/log/sum/max and
#    string(::Float64) differ from Rust. Rust `sort_by` is stable like `sort!`; replicate `sort_unstable*` where ties matter.
#  · usize → Int, u64 bit math → UInt64, integer `as` narrowing → `x % T`. BTreeMap → Dict + sorted keys.
#  · No runtime state at load (devices, env, clocks): precompiled consts are baked into the sysimage. Use lazy Refs.
#  · API types (Data Model Train TrainingReport RecipeApi) keep `_`-prefixed fields: API overloads getproperty so
#    scripts chain like Rust (`recipe.data("x").target("y")`); internal code reads `x._field`.
#  · Kernel templates come from build.jl (BLD.templates) as TPL["amd:default"], TPL["nvidia:-f32"], TPL["amd:gfx11-f16"], … (backend:Rust mapping key); no CPU target.
module Recipe
using Libdl, TOML, Printf, Mmap

# ==== SHARED ====
const ROOT = @__DIR__
const LL = joinpath(ROOT, "amd-nv-cpu.ll")
include_dependency(LL); include_dependency(joinpath(ROOT, "Project.toml"))
const CFG = TOML.parsefile(joinpath(ROOT, "Project.toml"))
# Rust build env RECIPE_<X> (build.rs key table) ↔ MR["key"] ([package.metadata.recipe]) or MT["key"] ([…train]).
const MR, MT = CFG["recipe"], CFG["train"]

"`@sec NAME begin … end`: section submodule that imports every name defined so far in Recipe."
macro sec(n, b)
	u = [Expr(:., x) for x in names(__module__; all=true) if x ∉ (n, :eval, :include) && !startswith(string(x), '#')]
	esc(Expr(:toplevel, Expr(:module, true, n, Expr(:block, :(using Libdl, TOML, Printf, Mmap),
		Expr(:using, Expr(:(:), Expr(:., :., :., :Recipe), u...)), b.args...))))
end

# ---- errors ----
struct RecipeError <: Exception; msg::String end
struct Panic <: Exception; msg::String end
Base.showerror(io::IO, e::Union{RecipeError,Panic}) = print(io, e.msg)
err(m) = throw(RecipeError(string(m)))
panic(m) = throw(Panic(string(m)))
require(c::Bool, m) = c ? nothing : err(m)
checked_add(a::Integer, b::Integer, role) = ((r, o) = Base.Checked.add_with_overflow(Int(a), Int(b)); o | (r < 0) ? err("$role overflows") : r)
checked_mul(a::Integer, b::Integer, role) = ((r, o) = Base.Checked.mul_with_overflow(Int(a), Int(b)); o | (r < 0) ? err("$role overflows") : r)

# ---- Rust numeric semantics ----
"Rust `x as T` for float x: truncate, saturate, NaN → 0."
rs(::Type{T}, x::AbstractFloat) where {T<:Integer} = isnan(x) ? zero(T) : x <= typemin(T) ? typemin(T) : x >= typemax(T) ? typemax(T) : unsafe_trunc(T, x)
for f in (:exp, :log, :log2, :log10, :log1p, :expm1, :exp2, :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :cbrt)
	g = Symbol(:l, f)
	@eval $g(x::Float64) = ccall(($(QuoteNode(f)), "libm.so.6"), Float64, (Float64,), x)
	@eval $g(x::Float32) = ccall(($(QuoteNode(Symbol(f, :f))), "libm.so.6"), Float32, (Float32,), x)
end
lpow(x::Float64, y::Float64) = ccall((:pow, "libm.so.6"), Float64, (Float64, Float64), x, y)
lpow(x::Float32, y::Float32) = ccall((:powf, "libm.so.6"), Float32, (Float32, Float32), x, y)
latan2(y::Float64, x::Float64) = ccall((:atan2, "libm.so.6"), Float64, (Float64, Float64), y, x)
lhypot(x::Float64, y::Float64) = ccall((:hypot, "libm.so.6"), Float64, (Float64, Float64), x, y)
"Rust `powi` (compiler-rt __powidf2: square-and-multiply, reciprocal last)."
function powi(a::T, b::Integer) where {T<:AbstractFloat}
	r, n = one(T), Int32(b)
	while true
		isodd(n) && (r *= a); n = div(n, Int32(2)); n == 0 && break; a *= a
	end
	b < 0 ? one(T) / r : r
end
"Rust `Iterator::sum` for floats: left fold from -0.0."
ssum(x) = foldl(+, x; init=-0.0)
ssum(f, x) = foldl((a, v) -> a + f(v), x; init=-0.0)
"Rust f64::max/min (NaN-ignoring)."
fmax(a, b) = isnan(a) ? b : isnan(b) ? a : max(a, b)
fmin(a, b) = isnan(a) ? b : isnan(b) ? a : min(a, b)
"Rust f64::round (ties away); Julia round(x) is Rust round_ties_even."
rround(x) = round(x, RoundNearestTiesAway)
logistic(v) = 1.0 / (1.0 + lexp(-v))
# shortest round-trip digits: value = 0.d × 10^p
function sdig(x::AbstractFloat)
	s = string(abs(x)); i = findfirst(c -> c == 'e' || c == 'f', s)
	m, e = i === nothing ? (s, 0) : (s[1:i-1], parse(Int, s[i+1:end]))
	a, b = split(m, '.'); d = a * b; p = length(a) + e; k = findfirst(!=('0'), d); d = rstrip(d[k:end], '0'); p -= k - 1
	# Ryu breaks an exact tie between two shortest strings to even; Rust's Dragon rounds the tie up.
	u = (big(10) // 1)^(p - length(d)); t = string(parse(BigInt, d) + 1)
	if 2 * (Rational{BigInt}(abs(x)) - parse(BigInt, d) * u) == u && parse(typeof(x), t * "e" * string(p - length(d))) == abs(x)
		p += length(t) - length(d); d = rstrip(t, '0')
	end
	(String(d), p)
end
_nf(x) = isnan(x) ? "NaN" : x > 0 ? "inf" : "-inf"
"Rust `{}` for f32/f64."
function fmtd(x::AbstractFloat)
	isfinite(x) || return _nf(x); g = signbit(x) ? "-" : ""; x == 0 && return g * "0"
	d, p = sdig(x); n = length(d)
	g * (p <= 0 ? "0." * "0"^-p * d : p >= n ? d * "0"^(p - n) : d[1:p] * "." * d[p+1:end])
end
"Rust `{:e}`."
fmte(x::AbstractFloat) = !isfinite(x) ? _nf(x) : x == 0 ? (signbit(x) ? "-0e0" : "0e0") :
	((d, p) = sdig(x); (signbit(x) ? "-" : "") * d[1:1] * (length(d) > 1 ? "." * d[2:end] : "") * "e" * string(p - 1))
"Rust `{:?}`."
fmtq(x::AbstractFloat) = !isfinite(x) ? _nf(x) : x != 0 && !(oftype(x, 1e-4) <= abs(x) < oftype(x, 1e16)) ? fmte(x) : ((s = fmtd(x)); occursin('.', s) ? s : s * ".0")
"Rust `{:.N}` (exact value, ties to even)."
fmtp(x::AbstractFloat, n::Integer) = isfinite(x) ? Printf.format(Printf.Format("%.$(n)f"), x) : _nf(x)
"Rust `{:?}` for strings: quoted with Rust escapes."
fmts(s::AbstractString) = "\"" * join(c == '"' ? "\\\"" : c == '\\' ? "\\\\" : c == '\n' ? "\\n" : c == '\t' ? "\\t" : c == '\r' ? "\\r" : c == '\0' ? "\\0" : string(c) for c in s) * "\""
"Copy an immutable struct with some fields replaced (Rust `S { f: v, ..s }`)."
with(x::T; kw...) where {T} = T((get(kw, f, getfield(x, f)) for f in fieldnames(T))...)
fieldeq(x::T, y::T) where {T} = all(getfield(x, f) == getfield(y, f) for f in fieldnames(T))

# ---- float encoding (recipe.rs 5607-5750, build.rs 8-150) ----
struct FL; sign::UInt8; exp::UInt8; man::UInt8 end
bias(l::FL) = (UInt64(1) << (l.exp - 1)) - 1
bits(l::FL) = l.sign + l.exp + l.man
power(e::Integer) = e > 1024 ? Inf : e < -1074 ? 0.0 : powi(2.0, e)
function pack(l::FL, v::Float64)
	e, m, z = Int(l.exp), Int(l.man), Int(bias(l))
	s = (reinterpret(UInt64, v) >> 63) << (e + m); el = (UInt64(1) << e) - 1; ml = UInt64(1) << m
	isnan(v) && return s | el << m | UInt64(1) << (m - 1)
	isinf(v) && return s | el << m
	v == 0 && return s
	a = abs(v); x = rs(Int64, floor(llog2(a)))
	if x < 1 - z
		q = rs(UInt64, round(a / power(1 - z) * Float64(ml)))
		return q == ml ? s | UInt64(1) << m : s | q
	end
	q = rs(UInt64, round((a / power(x) - 1.0) * Float64(ml)))
	q == ml && (q = UInt64(0); x += 1)
	x + z >= Int(el) ? s | el << m : s | UInt64(x + z) << m | q
end
function unpack(l::FL, b::UInt64)
	e, m, z = Int(l.exp), Int(l.man), Int(bias(l)); el = (UInt64(1) << e) - 1; ml = UInt64(1) << m
	x = b >> m & el; q = b & (ml - 1)
	r = x == el ? (q == 0 ? Inf : NaN) : x == 0 ? (q == 0 ? 0.0 : power(1 - z) * Float64(q) / Float64(ml)) :
		power(Int(x) - z) * (1.0 + Float64(q) / Float64(ml))
	b >> (e + m) != 0 ? -r : r
end
struct FF; arithmetic::FL; storage::FL end
native(s, e, m) = (l = FL(s, e, m); FF(l, l))
computed(e, m) = FF(FL(1, e, m), FL(1, 11, 52))
const F8, F16, F32, F64, B16 = native(1, 5, 2), native(1, 5, 10), native(1, 8, 23), native(1, 11, 52), native(1, 8, 7)
const T32 = FF(FL(1, 8, 10), FL(1, 8, 23))
bytes(f::FF) = cld(Int(bits(f.storage)), 8)
pack(f::FF, v::Float64) = pack(f.storage, unpack(f.arithmetic, pack(f.arithmetic, v)))
unpack(f::FF, b::UInt64) = unpack(f.arithmetic, pack(f.arithmetic, unpack(f.storage, b)))
struct IF; bits::UInt8 end
bytes(f::IF) = cld(Int(f.bits), 8)
pack(f::IF, v::Float64) = (n = Int(f.bits); reinterpret(UInt64, clamp(rs(Int64, round(v)), -(1 << (n - 1)), (1 << (n - 1)) - 1)) & ((UInt64(1) << n) - 1))
unpack(f::IF, b::UInt64) = (n = 64 - Int(f.bits); Float64(reinterpret(Int64, b << n) >> n))
"Rust enum Compute: kind ∈ :f :fp :int :bf :tf; `f` for floats, `i` for :int (unused field canonical)."
struct Compute; kind::Symbol; f::FF; i::IF end
Compute(k::Symbol, f::FF) = Compute(k, f, IF(0))
Compute(i::IF) = Compute(:int, F64, i)
const FP8, FP16, FP32, FP64 = Compute(:fp, F8), Compute(:fp, F16), Compute(:fp, F32), Compute(:fp, F64)
const INT1, INT4, INT8 = Compute(IF(1)), Compute(IF(4)), Compute(IF(8))
const BF16, TF32 = Compute(:bf, B16), Compute(:tf, T32)
bytes(c::Compute) = c.kind == :int ? bytes(c.i) : c.kind == :f ? bytes(F64) : bytes(c.f)
pack(c::Compute, v::Float64) = c.kind == :int ? pack(c.i, v) : pack(c.f, v)
unpack(c::Compute, b::UInt64) = c.kind == :int ? unpack(c.i, b) : unpack(c.f, b)
optimizer_epsilon(c::Compute, v) = (r = unpack(c, pack(c, v)); c.kind != :int && r == 0 ? unpack(c.f.arithmetic, UInt64(1) << c.f.arithmetic.man) : r)
below_one(c::Compute, v) = (r = unpack(c, pack(c, v)); c.kind != :int && r >= 1 ? unpack(c.f.arithmetic, pack(c.f.arithmetic, 1.0) - 1) : r)
saved_fields(c::Compute) = c.kind == :int ? ("int", (c.i.bits, 0x00, 0x00, 0x00)) :
	(string(c.kind), (bits(c.f.storage), c.f.arithmetic.exp, c.f.arithmetic.man, c.f.storage.man))
_pick(cs, v) = (i = findfirst(c -> saved_fields(c)[2] == v, cs); i === nothing ? nothing : cs[i])
"Compute::saved(family, [bits, exp, man, storage_man]) → Compute or nothing."
function saved(fam::AbstractString, v::NTuple{4,UInt8})
	b, e, m, sm = v
	fam == "f" ? (b == bits(F64.storage) && sm == F64.storage.man && e != 0 && m != 0 && Int(e) + Int(m) < 64 ? Compute(:f, computed(e, m)) : nothing) :
	fam == "fp" ? _pick((FP8, FP16, FP32, FP64), v) : fam == "int" ? _pick((INT1, INT4, INT8), v) :
	fam == "bf" ? _pick((BF16,), v) : fam == "tf" ? _pick((TF32,), v) : nothing
end
label(c::Compute) = c.kind == :f ? "f($(c.f.arithmetic.exp),$(c.f.arithmetic.man))" : c.kind == :int ? "int$(c.i.bits)" : "$(c.kind)$(bits(c.f.storage))"

# ---- config (recipe.rs 10398-10587) ----
number(n, v) = (x = Float64(v); isfinite(x) && x > 0 ? x : err("$n must be finite and positive"))
fraction(n, v) = (x = number(n, v); require(x <= 1, "$n must not exceed one"); x)
cnt(n, v) = v isa Integer && v >= 0 ? Int(v) : err("invalid $n: invalid digit found in string")
natural(n, v) = (x = cnt(n, v); require(x != 0, "$n must be positive"); x)
arguments(a, b) = [Float64(a), Float64(b), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
struct Config
	kmeans_iterations::Int; svm_iterations::Int; svm_rate::Float64; svm_regularization::Float64; svm_epsilon::Float64
	tree_depth::Int; tree_min_rows::Int; forest_feature_fraction::Float64; bayes_prior_precision::Float64
	bayes_noise_variance::Float64; bayes_variance_epsilon::Float64; boost_iterations::Int; boost_rate::Float64
	catboost_prior::Float64; catboost_borders::Int; xgboost_regularization::Float64; xgboost_min_gain::Float64
	lightgbm_bins::Int; lightgbm_leaves::Int; quantization_block::Int; surrogate_epochs::Int; surrogate_width::Int
	surrogate_rate::Float64; initial::Float64; beta1::Float64; beta2::Float64; epsilon::Float64; decay::Float64
	progress_refresh_hz::Int; random_seed::Int; activation::NTuple{8,Float64}; precision::Compute
end
"Config::load. Field order above is Rust's; `with(c; precision=p)` replaces fields."
function load_config()
	N(n, k) = natural(n, MT[k]); R(n, k) = number(n, MT[k])
	Config(N("kmeans iterations", "kmeans-iterations"), N("SVM iterations", "svm-iterations"), R("SVM learning rate", "svm-learning-rate"),
		R("SVM regularization", "svm-regularization"), R("SVM epsilon", "svm-epsilon"), N("tree depth", "tree-depth"),
		N("tree minimum rows", "tree-min-rows"), fraction("forest feature fraction", MT["forest-feature-fraction"]),
		R("Bayes prior precision", "bayes-prior-precision"), R("Bayes noise variance", "bayes-noise-variance"),
		R("Bayes variance epsilon", "bayes-variance-epsilon"), N("boost iterations", "boost-iterations"),
		fraction("boost learning rate", MT["boost-learning-rate"]), R("CatBoost ordered prior", "catboost-ordered-prior"),
		N("CatBoost border count", "catboost-border-count"), R("XGBoost L2 regularization", "xgboost-l2-regularization"),
		R("XGBoost minimum gain", "xgboost-minimum-gain"), N("LightGBM histogram bins", "lightgbm-histogram-bins"),
		N("LightGBM leaves", "lightgbm-leaves"), N("quantization block weights", "quantization-block-weights"),
		N("surrogate epochs", "surrogate-epochs"), N("surrogate width", "surrogate-width"), R("surrogate rate", "surrogate-rate"),
		R("initial weight", "initial-weight"), R("AdamW beta1", "adamw-beta1"), R("AdamW beta2", "adamw-beta2"),
		R("AdamW epsilon", "adamw-epsilon"), R("AdamW weight decay", "adamw-weight-decay"), N("progress refresh Hz", "progress-refresh-hz"),
		N("random seed", "random-seed"), (R("leak slope", "leak-slope"), R("PReLU slope", "prelu-slope"), R("ELU alpha", "elu-alpha"),
		R("SELU alpha", "selu-alpha"), R("SELU scale", "selu-scale"), R("GELU scale", "gelu-scale"), R("GELU cubic", "gelu-cubic"),
		R("Huber threshold", "huber-threshold")), FP64)
end

# ---- globals (recipe.rs 6894-6962) ----
const RUN = Threads.Atomic{UInt64}(0)
const INTERRUPTED, INTERRUPT_CHECKPOINTED = Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(false)
const SIGINT, INTERRUPTED_EXIT = 2, 130
const DEBUG_LOG_PATH = joinpath(ROOT, "recipe.log")
const DEBUG_LOG = Ref{Any}(nothing); const DEBUG_LOCK = ReentrantLock()
function debug(m)
	haskey(ENV, "RECIPE_DEBUG") || return
	lock(DEBUG_LOCK) do
		DEBUG_LOG[] === nothing && (DEBUG_LOG[] = try open(DEBUG_LOG_PATH, "w") catch e; err("cannot open $DEBUG_LOG_PATH: $(sprint(showerror, e))") end)
		println(DEBUG_LOG[], m); flush(DEBUG_LOG[])
	end
end
@enum Backend::UInt8 Amd=1 Nvidia=2
const VP = Ptr{Cvoid}

# ---- model DSL types (recipe.rs 6965-7103) ----
@enum Activation::UInt8 ALinear ACos AExp ALog ALn AHuber ATan ARelu ALeak ASigmoid ATanh ASelu AGelu ASilu AElu APrelu
@enum BlockNormalization::UInt8 NBatch NLayer
"Rust Residual: kind :layer(a) :conv(a=filters, b=kernel) :activation(act)."
struct Residual; kind::Symbol; a::Int; b::Int; act::Activation end
struct Estimator; fit::Function; validate::Function; param::Int; name::String end
Base.:(==)(x::Estimator, y::Estimator) = x.param == y.param && x.name == y.name
"Rust Operation: kind :layer(a) :conv(a,b) :pool(a) :estimator(est) :attention(a) :rnn(a) :gru(a) :lstm(a) :residual(parts) :moe(a=top_k, parts) :perceptron(a)."
struct Operation; kind::Symbol; a::Int; b::Int; est::Union{Nothing,Estimator}; parts::Vector{Residual} end
Operation(k::Symbol, a=0, b=0) = Operation(k, a, b, nothing, Residual[])
struct Block; operation::Operation; activation::Activation; normalization::Union{Nothing,BlockNormalization}; quantization::UInt16; profile::Bool end
Base.:(==)(x::T, y::T) where {T<:Union{Operation,Block}} = fieldeq(x, y)
struct LossFunction; c::UInt8 end   # 0 mse 1 rmse 2 huber 3 mae 4 bce/ce 6 focal
struct Metric; c::UInt8 end         # 0 Run 1 Loss 2 R2 3 Time 4 Epoch 5 blck 6 atvn 7 norm 8 tok 9 quant 10 tile
struct Model; _blocks::Vector{Block}; _loss::LossFunction; _quantization::UInt16 end
struct Auto end; struct ZScore end; struct Batch end; struct Adamw end; struct RecipeApi end
const CHAR_IDS = collect("\t\n !\"#\$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¦±€")

# ---- graph types (recipe.rs 9278-9408) ----
struct Shape; channels::Int; length::Int end
elements(s::Shape) = s.channels * s.length
@enum Primitive::Int32 PContraction=0 PPool=2 PAttention=4 PScan=5 PElementwise=6 PNormalize=8 PPredictor=9
@enum ScalarOpcode::Int32 SAdd=0 SConstant=1 SParameter=2 SSubtract=3 SMultiply=4 SDivide=5 SAbsolute=6 SExp=7 SLog=8 SSin=10 SCos=11 STanh=12 SGreater=13 SStraightThrough=14 SSelect=15
@enum PredictorOpcode::Int32 POFeature PORow POConstant POLoad POStore PODuplicate POAdd POSubtract POMultiply PODivide POGreater POChoose PONearest POAffine
struct ScalarProgram; v::Vector{Float64} end
ScalarProgram() = ScalarProgram(Float64[])
op(p::ScalarProgram, o::ScalarOpcode, l, r) = (n = Float64(length(p.v) ÷ 3); push!(p.v, Float64(Int32(o)), l, r); n)
constant(p::ScalarProgram, x) = op(p, SConstant, x, 0.0)
"Both branches are straight-line SSA: drop the untaken one by selection, never by multiply masking (0*inf = NaN)."
select(p::ScalarProgram, c, x) = op(p, SSelect, c, x)
choose(p::ScalarProgram, c, y, n) = (o = constant(p, 1.0); i = op(p, SSubtract, o, c); a = select(p, c, y); b = select(p, i, n); op(p, SAdd, a, b))
unary(p::ScalarProgram, o::ScalarOpcode, x) = op(p, o, x, 0.0)
mutable struct Node
	op::Primitive; source::Int32; second::Int32; input::Shape; output::Shape; offset::Int; parameters::Int
	argument::Vector{Float64}; program_offset::Int; program_count::Int; block_index::Int; block_kind::String
end
node_identity(n::Node, i) = "block $(n.block_index) $(n.block_kind), node $i $(string(n.op)[2:end]), input $(n.input.channels)x$(n.input.length), output $(n.output.channels)x$(n.output.length), offset=$(n.offset) count=$(n.parameters), source=$(n.source)"
mutable struct TrainingState; moments::Vector{Float64}; variances::Vector{Float64}; best_loss::Vector{Float64}; trained_samples::Vector{UInt64}; epoch::Int; training_rows::Int end
TrainingState() = TrainingState(Float64[], Float64[], Float64[], UInt64[], 0, 0)
struct StorageFormat; code::UInt16 end
mutable struct StoredWeight; format::StorageFormat; count::Int; bytes::Vector{UInt8}; codebook::Vector{Float64}; arithmetic::Vector{Float64} end
"Rust Graph (Clone → deepcopy)."
mutable struct Graph
	nodes::Vector{Node}; parameters::Vector{Float64}; frozen::Vector{UInt8}; programs::Vector{Float64}
	stored::Vector{Union{Nothing,StoredWeight}}; input::Shape; output::Shape; source::Int32; state::TrainingState
	block_index::Int; block_kind::String
end
Graph(s::Shape) = Graph(Node[], Float64[], UInt8[], Float64[], Union{Nothing,StoredWeight}[], s, s, Int32(-1), TrainingState(), 0, "")
refresh_storage(g::Graph, c::Config) = encode_graph_storage(g, c)
struct PredictorProgram; code::Vector{Float64}; locals::Int; stack::Int; table::Vector{Float64} end

# ---- data types (recipe.rs 15475-15531) ----
const DataSchema = Vector{Tuple{String,String}}
mutable struct Prepared
	samples::Vector{Float64}; targets::Vector{Float64}; target_width::Int; rows::Int; source_rows::Int; features::Int
	schema::DataSchema; sequence::Union{Nothing,Shape}; target_categorical::Bool; norm_mean::Vector{Float64}
	norm_scale::Vector{Float64}; identities::Vector{UInt64}; fitted::Vector{PredictorProgram}
end
mutable struct Data
	_sources::Vector{String}; _tests::Vector{String}; _autoregressive::Bool; _target::Vector{String}; _exclusions::Vector{String}
	_broadcast::Bool; _normalize::Bool; _split::Float64; _prepared::Union{Nothing,Prepared,RecipeError}  # OnceLock<Result<Prepared>>
end
mutable struct Table; name::String; headers::Vector{String}; rows::Vector{Vector{String}} end
"Rust FeatureType: kind :numeric, :categorical(cats), :text(n)."
struct FeatureType; kind::Symbol; cats::Vector{String}; n::Int end

# ---- schedule / native IR types (recipe.rs 1951-2283, 4790-4800, 5136-5151, 10419-10456) ----
struct Tile; m::UInt32; n::UInt32; k::UInt32 end
struct NativeContractionTiles; forward::Tile; gradient::Tile; previous::Tile; gradient_shape::Tile; parameters::Int end
struct NativeContractionShapes; forward::Tile; gradient::Tile; previous::Tile; parameters::Int end
mutable struct NativeSchedule
	matrix::Bool; block::UInt32; tile::Tile; register_m::UInt32; register_n::UInt32; register_count::UInt32; fragment_k::UInt32
	chunk_k::UInt32; chunk_values::UInt32; chunk_bias_values::UInt32; scratch_base::Int32; shared_values::UInt32
	contractions::Vector{Union{Nothing,NativeContractionTiles}}; attention::Vector{Union{Nothing,Tile}}
end
struct NativeLayout; values::Vector{Int}; contexts::Vector{Int}; adjoints::Vector{Int}; values_bytes::Int; contexts_bytes::Int; adjoints_bytes::Int end
"Rust BackendTarget without Cpu: backend Amd (gfx…) or Nvidia (sm_…)."
struct BackendTarget; backend::Backend; architecture::String end
struct NativePrecision; model::Compute; state::Compute; source::String; model_type::String; state_type::String; epoch_layout::String end
mutable struct NativeArtifact; backend::BackendTarget; layout::NativeLayout; precision::NativePrecision; artifact::Vector{UInt8}; path::String; storage::Vector{UInt8} end
const NATIVE_FORWARD_SYMBOL, NATIVE_EPOCH_SYMBOL, NATIVE_MODEL_LOAD_SYMBOL = "recipe_model_forward", "recipe_model_epoch", "recipe_model_load"
const NATIVE_FORWARD_LAYOUT, NATIVE_MODEL_LOAD_LAYOUT = "888844", "884"
const NATIVE_EPOCH_LAYOUT_FP64, NATIVE_EPOCH_LAYOUT_FP32 = "8888888888884488888884", "8888888888884444444444"
const NATIVE_SCALAR_PARTITIONS, NATIVE_SCRATCH_ROW_VALUES = 4096, 4
const NATIVE_SPLIT_SPAN = natural("contraction split span", MR["contraction-split-span"])
const NATIVE_MATRIX_SPLIT_SPAN = natural("contraction matrix split span", MR["contraction-matrix-split-span"])
const NATIVE_K_PARTITIONS = natural("contraction K partitions", MR["contraction-k-partitions"])
"NativePrecision::new: (template suffix, model type, state, state type, epoch layout) per Compute."
function native_precision(c::Compute)
	L, S = NATIVE_EPOCH_LAYOUT_FP32, FP32
	t = c.kind == :f ? ("-f", "double", FP64, "double", NATIVE_EPOCH_LAYOUT_FP64) :
		c == FP64 ? ("default", "double", FP64, "double", NATIVE_EPOCH_LAYOUT_FP64) : c == FP32 ? ("-f32", "float", S, "float", L) :
		c == FP16 ? ("-f16", "half", S, "float", L) : c == FP8 ? ("-f8", "i8", S, "float", L) : c == BF16 ? ("-bf16", "i16", S, "float", L) :
		c == TF32 ? ("-tf32", "float", S, "float", L) : c == INT8 ? ("-int8", "i8", S, "float", L) :
		c == INT4 ? ("-int4", "i8", S, "float", L) : c == INT1 ? ("-int1", "i8", S, "float", L) :
		err("$(label(c)) has no native precision composition")
	NativePrecision(c, t[3], t[1], t[2], t[4], t[5])
end
@enum NativeMatrix::UInt8 Gfx11 Gfx12
mutable struct NodePlan; node::Node; value::Int; context::Int; adjoint::Int; stored::Union{Nothing,StoredWeight}; storage_offset::Int end
mutable struct NativeModelIr; graph::Graph; layout::NativeLayout; precision::NativePrecision; rows::Int; schedule::NativeSchedule; plans::Vector{NodePlan}; storage_bytes::Int end
struct ModelPointers; source::String; second::String; value::String; context::String; delta::String; weights::String; source_adjoint::String; second_adjoint::String end
Base.@kwdef struct PartitionedLoop; count::Int; partitions::Int; columns::Int; value_type::String; pointer_type::String; scratch::String; zero::String; gradients::Vector{Tuple{Int,String}} end
# program_ir contexts (recipe.rs 141-172, 959-977, 1407-1550); `literal(value::Float64, ty::String)::String`.
@enum NormalizeMode::UInt8 MBatch MLayer MRms MEvaluation
Base.@kwdef struct ScalarContext; value_type::String; pointer_type::String; alignment::Int; first::String; second::String; weights::String; prefix::String; literal::Function end
struct ScalarForward; code::String; value::String end
struct ScalarReverse; code::String; first_adjoint::String; second_adjoint::String; parameter_adjoint::Vector{Pair{Int,String}} end  # sorted by key
Base.@kwdef struct PredictorContext; value_type::String; pointer_type::String; alignment::Int; input::String; row::String; features::Int; weights::String; offset::Int; parameters::Int; prefix::String; literal::Function end
struct PredictorForward; code::String; value::String end
Base.@kwdef struct NormalizeContext; value_type::String; pointer_type::String; alignment::Int; source_value::String; context::String; rows::String; channels::Int; length::Int; mode::NormalizeMode; prefix::String end
struct NormalizeFragment; code::String; value::String end
Base.@kwdef struct NormalizeReverseContext; value_type::String; pointer_type::String; alignment::Int; state_type::String; state_zero::String; context::String; rows::String; channels::Int; length::Int; mode::NormalizeMode; prefix::String end
struct NormalizeReverseFragment; code::String; contribution::String end

# ---- bundle / train / report types (recipe.rs 6019-6051, 18707-18717, 19511-19560) ----
mutable struct StoredGraph; graph::Graph; model::Model; precision::Compute; inputs::Vector{String}; outputs::Vector{String}; norm_mean::Vector{Float64}; norm_scale::Vector{Float64}; target_min::Float64; target_span::Float64; bn_stats::Vector{Float64}; artifact::String end
mutable struct SemanticGraph; model::Model; precision::Compute; input::Shape; output::Shape; inputs::Vector{String}; outputs::Vector{String}; tensors::Vector{StoredWeight}; predictors::Vector{PredictorProgram}; frozen::Vector{UInt8}; state::TrainingState; norm_mean::Vector{Float64}; norm_scale::Vector{Float64}; target_min::Float64; target_span::Float64; bn_stats::Vector{Float64}; artifact::String end
mutable struct Train; _epochs::Int; _learning_rate::Float64; _log_metrics::Vector{Metric}; _stop::Union{Nothing,Float64}; _resume::Union{Nothing,String}; _save::Union{Nothing,String}; _seed::Union{Nothing,Int}; _precision::Compute end
struct Metrics; run::UInt64; epoch::Int; loss::Union{Nothing,Float64}; r2::Union{Nothing,Float64}; seconds::Float64; checkpoint::Bool; evaluation::Bool end
struct TrainingReport; _initial_loss::Float64; _final_loss::Float64; _initial_predictions::Vector{Float64}; _predictions::Vector{Float64}; _r2::Float64; _tile::Tile; _schedule::String; _run::UInt64; _epoch::Int; _seconds::Float64; _epoch_seconds::Float64 end
struct TargetScale; minimum::Float64; span::Float64 end

# ---- kernel templates (build.rs): TPL["amd:default"], TPL["nvidia:-f16"], TPL["amd:gfx12-int8"], … ----
@sec BLD begin
	include("build.jl")
end
const TPL = BLD.templates(MR, read(LL, String))

# ---- cross-section stubs: provider section, then Rust origin. Implement as `Recipe.name(…) = …`. ----
# PIR (mod program_ir): code::Vector{Float64}; contexts above; throw RecipeError for Rust EmitError (same Display text).
function emit_scalar_forward end          # (code, ::ScalarContext) → ScalarForward
function emit_scalar_reverse end          # (code, ::ScalarContext, incoming::String, …Rust args) → ScalarReverse
function emit_predictor_forward end       # (code, locals, ::PredictorContext) → PredictorForward
function emit_normalize end               # (::NormalizeContext, element) → NormalizeFragment
function emit_normalize_reverse_stats end # (::NormalizeReverseContext, delta, output_value, …) → String
function emit_normalize_reverse end       # (::NormalizeReverseContext, element, delta, …) → NormalizeReverseFragment
# IRA (NativeLayout, BackendTarget, NativeModelIr part 1, templates, emit_fixed_primitives)
function native_layout end                # NativeLayout::for_graph(graph, rows, precision)
function layout_storage end               # NativeLayout::storage(layout) → Vector{UInt8}
function native_model_ir end              # NativeModelIr::from_graph(graph, rows, precision, …) → NativeModelIr
function target_backend end               # BackendTarget::backend
function artifact_extension end           # BackendTarget::artifact_extension → "hsaco" | "ptx"
function validate_target end              # BackendTarget::validate
function backend_template end             # (backend, ::NativePrecision, matrix::Union{Nothing,NativeMatrix}, …) → String (from TPL)
function pointer_type end                 # (backend) → "ptr addrspace(1)"…
function barrier end                      # (backend) → String
function ptr_gep end                      # (backend, base, offset, name) → String
function definition_span end              # (ir, name) → Union{Nothing,Tuple{Int,Int}}
function strip_definition end             # (ir, name) → String
function prune_internal_definitions end   # (ir) → String
function emit_fixed_primitives end        # (::NativeModelIr, backend, matrix::Bool, …) → String
function align end                        # (value, boundary) → Int
# QIR (mod quantized): QuantOps generic over host decode (QIR.HostQuantOps) and IR emission (QIR.NativeQuantOps).
function dequant_iq end; function dequant_iq1 end; function dequant_q45k end; function dequant_q6k end; function dequant_scalar end
function dequant_q2k end; function dequant_q3k end; function dequant_q8k end; function dequant_iq4 end; function dequant_nf4 end
# IRB (NativeModelIr part 2, loss IR, loops)
function emit_model end                   # NativeModelIr::emit(ir, backend, …) → String (full module IR)
function emit_pointers end                # NativeModelIr::emit_pointers(ir, backend, index, …) → ModelPointers
function emit_partitioned_loop end        # (ir::IOBuffer|String, index, name, ::PartitionedLoop, …)
function emit_fixed_loop end              # (ir, index, name, …)
function accumulate_owned end             # (target, value, ty, pointer, prefix) → String
function type_literal end                 # (ty, value) → String
function native_literal end               # (precision::Compute, ty, value) → String
function normalize_mode end               # (value::Float64) → NormalizeMode
function integer_argument end             # (value::Float64, role) → Int32
function alignment end                    # (ty) → Int
function loss_threshold end               # (precision, ty) → String
function emit_loss_value end; function emit_loss_gradient end
# TOOL (toolchain, compile pipeline, tiles, geometry)
function compile_model end                # (::BackendTarget, graph, precision, rows, schedule, …) → NativeArtifact
function compile_native_artifact end      # (target, source, output, …) → Vector{UInt8}
function native_artifact_key end          # (target, ir) → String
function native_artifact_directory end    # (key) → String
function kernel_resources end             # (diagnostic) → Vector of (name, registers, scalars, occupancy)
function native_amd_compiler end; function native_nvidia_compiler end; function native_amd_library end
function native_contraction_shapes end    # (graph, rows) → Vector{Union{Nothing,NativeContractionShapes}}
function native_attention_shared_values end; function native_attention_tile end; function native_attention_tiles end
function native_tiles end; function native_contraction_partial_per_chunk end; function native_contraction_tile end
function native_contraction_shared_values end; function native_gradient_values end
function geometry end; function amd_resources end; function nvidia_resources end   # Rust geometry, amd, nvidia
# BUNDLE (mod bundle: .ogdl save/load)
function bundle_artifact_key end          # bundle::artifact_key(model, schema, precision, graph, target) → String
function load_semantic end                # (path) → (DataSchema, Vector{SemanticGraph})
function save_semantic end; function save_semantic_graphs end; function restore end
function run_infer end                    # (path, input, forward::Function(::SemanticGraph, samples)) → Vector{Float64}
# QUANT (host codecs, IQ grids, StorageFormat)
function quantization end                 # (code::UInt16) → name String
function fp16 end; function unfp16 end; function put_half end; function half end
function valid end; function spec end; function encode end; function unavailable end; function selection end
function tensor end; function decompress end; function nf4_codebook end
# API (Model DSL, Recipe, losses, metrics, Train, reports, CLI)
function julia_main end                   # ()::Cint — cli.rs main: recipe [--device <name>] <source.jl> [export]
function loss_name end; function loss_value end   # LossFunction::name / ::value(l, prediction, target, threshold)
function description end                  # Model::description(model, metrics)
function model_loss end; function coefficient end; function predicted_char end
# GRAPH (lowering)
function encode_graph_storage end         # (graph, config)
function compile end                      # (model, prepared, targets, …) → Graph (recipe.rs 9436)
function materialize_saved_graph end; function append_graph end; function lower_block end
function stored_graph end; function output_bias_offset end; function initialize_graph end
# GPU (NativeTape, Buffer, Cuda, Hsa, NativeProgram, devices)
function native_tape end                  # NativeTape::new(graph, samples, targets, device, precision, loss)
function forward end; function epoch end; function predictions end; function inject_bn_stats end; function extract_bn_stats end
function observe end; function advance end; function weights end; function optimizer_state end; function capture end
function tape_tile end; function tape_schedule end   # NativeTape::tile / ::schedule
function checkpoint end; function graph_rows_buffer end; function node_context end; function structural end; function narrow end
function devices end; function device end; function selected_gpu end; function print_devices end; function native_target_label end
# ML (classical estimators, predictor programs, surrogate)
function fit_svm end; function fit_forest end; function fit_bayes end; function fit_xgboost end; function fit_lightgbm end
function fit_catboost end; function fit_kmeans end; function fit_knn end
function valid_estimator end; function positive_estimator end; function cluster_estimator end; function neighbor_estimator end
function evaluate end                     # PredictorProgram::evaluate(p, row, query) → Float64
function fit_surrogate end; function surrogate_model end; function graph_inputs end; function predict_rows end
function parallel_map end; function next_random end; function distance end; function nearest end
# DATAA (Data API, sample builder, images, prepare)
function prepare end                      # (data) → Prepared (cached in data._prepared)
function prepare_data end; function resolve_path end; function data_sources end   # IntoDataSources
# DATAB (decoders, features, shuffle)
function decode_tables end; function parse_table end; function infer_feature end; function categories end
function encode_feature end; function shuffle_rows end; function zip_entries end; function zlib_inflate end; function inflate end

# ==== PIR ====
@sec PIR begin
# recipe.rs 4-1950 (mod program_ir): scalar/predictor/normalize LLVM IR emission. Budget 58,000. Parity: IR text.
end

# ==== IRA ====
@sec IRA begin
# recipe.rs 1951-2521, 3506-4113: NativeLayout, BackendTarget (no CPU), NativeModelIr::from_graph, templates, emit_fixed_primitives. Budget 47,000. Parity: IR text.
end

# ==== QIR ====
@sec QIR begin
# recipe.rs 2522-3505 (mod quantized): QuantOps, dequant_* over HostQuantOps / NativeQuantOps. Budget 36,000. Parity: IR text, host bits.
end

# ==== IRB ====
@sec IRB begin
# recipe.rs 4114-5234: pointers, quantized decoders, model load, emit, loss/seed, AdamW, loss IR, loops. Budget 36,000. Parity: IR text.
end

# ==== TOOL ====
@sec TOOL begin
# recipe.rs 5235-5606, 14872-15439: artifact cache, clang/llc calls (no CPU), compile_model, tiles, geometry. Budget 26,000. Parity: commands, tiles.
# Calls stubs validate_target artifact_extension native_model_ir emit_model loss_name narrow integer_argument, and
# layout_storage(::NativeModelIr) for Rust NativeModelIr::storage. u32 math runs in Int128, usize math in UInt64 (release
# build: next_multiple_of wraps); u32 results are UInt32. Resources = (shared, max_block) and Geometry = (groups, block)
# NamedTuples; Rust Geometry::threads is Recipe.TOOL.gthreads. compile_native_artifact(…; x) takes the runner (Cmd, role, key) → stderr.
const SERIAL = Threads.Atomic{Int}(0)
W(x) = Int128(x)
U(x) = UInt64(x)
u(x, m) = x > 0xffffffff ? err(m) : Int(x)
cm(a, b, r) = ((p, o) = Base.mul_with_overflow(U(a), U(b)); o ? err("$r overflows") : p)
ca(a, b, r) = ((p, o) = Base.add_with_overflow(U(a), U(b)); o ? err("$r overflows") : p)
nw(v, r) = narrow(Int(min(U(v), U(typemax(Int)))), r)
ioe(e) = e isa Base.IOError ? (c = -e.code; "$(Libc.strerror(c)) (os error $c)") : e isa SystemError ? "$(Libc.strerror(e.errnum)) (os error $(e.errnum))" : sprint(showerror, e)
io(f, m) = try f() catch e; e isa RecipeError ? rethrow() : err("$m: $(ioe(e))") end
mr(k, m) = (v = get(MR, k, nothing); v === nothing ? err(m) : string(v))

function Recipe.native_artifact_directory(k)
	require(!isempty(k) && k ∉ (".", "..") && !occursin('/', k) && !occursin('\\', k), "native artifact key is not a single path component")
	h = get(ENV, "HOME", nothing); h === nothing && err("home directory is absent")
	joinpath(h, ".cache", "recipe", "native", k)
end
"FNV-1a over (u64 LE length, bytes) of the version tag, the target label and the IR."
function Recipe.native_artifact_key(t::BackendTarget, ir)
	h = 0xcbf29ce484222325
	for p in ("recipe-native-v2", t.architecture, ir), b in Iterators.flatten((reinterpret(UInt8, [htol(U(ncodeunits(p)))]), codeunits(p)))
		h = (h ⊻ b) * 0x00000100000001b3
	end
	"recipe-native-" * string(h, base=16, pad=16)
end
"Rust `{:?}` of a Command."
cq(c::Cmd) = join(fmts.(c.exec), ' ')
lossy(b) = (s = String(b); isvalid(s) ? s : map(c -> isvalid(c) ? c : '�', s))
"Run a compiler; return its trimmed diagnostic stream (resource remarks arrive there)."
function ncmd(c::Cmd, r, k)
	debug("native compiler key=$k role=$r command=$(cq(c))")
	e = IOBuffer()
	p = try run(pipeline(ignorestatus(c); stdin=devnull, stdout=devnull, stderr=e)) catch x; err("cannot start $r: $(ioe(x))") end
	d = String(strip(lossy(take!(e))))
	success(p) ? d : err("$r failed: $d")
end
"`occupancy` is resident waves per SIMD; zero means the cooperative grid could never complete."
mutable struct KernelResources; name::String; registers::UInt32; scalars::UInt32; occupancy::UInt32 end
const KF = Dict("VGPRs" => :registers, "TotalSGPRs" => :scalars, "Occupancy [waves/SIMD]" => :occupancy)
pu(v) = UInt32(occursin(r"^\+?[0-9]+\z", v) ? something(tryparse(UInt32, lstrip(v, '+')), 0) : 0)
"Parse `-Rpass-analysis=kernel-resource-usage` remarks; unrecognised text yields no entries."
function Recipe.kernel_resources(d::AbstractString)
	f = KernelResources[]
	for l in split(d, '\n')
		l = chopsuffix(l, "\r"); occursin("kernel-resource-usage", l) || continue
		i = findlast(" [-", l); b = i === nothing ? l : l[1:prevind(l, first(i))]
		j = findlast(':', b); j === nothing && continue
		h, v = b[1:prevind(b, j)], String(strip(b[j+1:end])); a = strip(h[something(findlast(':', h), 0)+1:end])
		a == "Function Name" ? push!(f, KernelResources(v, 0, 0, 0)) :
			(s = get(KF, a, nothing)) !== nothing && !isempty(f) && setfield!(f[end], s, pu(v))
	end
	f
end
Recipe.native_amd_compiler() = mr("hsa-compiler", "AMD native compiler is unavailable")
Recipe.native_nvidia_compiler() = mr("nvidia-compiler", "NVIDIA native compiler is unavailable")
Recipe.native_amd_library(n) = mr("hsa-$n-library", "AMD native $n library is unavailable")
const LB = ("-Xclang", "-mlink-builtin-bitcode", "-Xclang")
function Recipe.compile_native_artifact(t::BackendTarget, s, o, bc, k; x=ncmd)
	a = t.architecture
	if t.backend == Amd
		# The resource remarks are the only route to the kernel's register allocation, which decides residency.
		c = [native_amd_compiler(), "-target", "amdgcn-amd-amdhsa", "-mcpu=$a", "-O3", "-nogpulib", "-Rpass-analysis=kernel-resource-usage"]
		for n in ("device", "clock", "abi", "finite", "math") push!(c, LB..., native_amd_library(n)) end
		d = mr("hsa-device-library-directory", "AMD native device library directory is unavailable")
		push!(c, LB..., joinpath(d, "oclc_isa_version_$(replace(a, r"^(gfx)*" => "")).bc"), s, "-o", o)
		return kernel_resources(x(Cmd(c), "AMD LLVM IR compiler", k))
	end
	c = native_nvidia_compiler(); v = mr("nvidia-device-library", "NVIDIA native device library is unavailable")
	bc === nothing && err("NVIDIA bitcode path is absent")
	p = "+" * mr("nvidia-ptx", "NVIDIA PTX version is unavailable")
	x(Cmd([c, "-target", "nvptx64-nvidia-cuda", "-march=$a", "-O2", "-emit-llvm", "-c", "-x", "ir", s, LB..., v, "-o", bc]), "NVIDIA LLVM IR compiler", k)
	# clang stamps its newest PTX ISA, so llc pins the configured one and every driver at or above it JIT-loads the artifact.
	g = mr("nvidia-ptx-generator", "NVIDIA PTX generator is unavailable")
	x(Cmd([g, "-march=nvptx64", "-mcpu=$a", "-mattr=$p", "-O2", bc, "-o", o]), "NVIDIA PTX generator", k)
	io(() -> write(o, push!(read(o), 0x00)), "cannot terminate native PTX artifact")
	KernelResources[]
end
function Recipe.compile_model(t::BackendTarget, g, p::Compute, l, rows, sc)
	validate_target(t)
	m = native_model_ir(g, rows, p, sc); a = t.architecture; e = artifact_extension(t)
	x = t.backend == Amd && m.schedule.matrix ? (startswith(a, "gfx11") ? Gfx11 : startswith(a, "gfx12") ? Gfx12 : nothing) : nothing
	ir = emit_model(m, t.backend, x, l)
	k = native_artifact_key(t, ir); d = native_artifact_directory(k)
	io(() -> mkpath(d), "cannot create native artifact directory")
	f = joinpath(d, "artifact.$e"); h = isfile(f)
	debug("native artifact key=$k target=$a arithmetic=$(label(m.precision.model)) loss=$(loss_name(l)) rows=$rows cache=$(h ? "hit" : "miss") path=$f")
	if !h
		st = joinpath(d, ".recipe-native-$(getpid())-$(Threads.atomic_add!(SERIAL, 1))")
		s, o, b = "$st.ll", "$st.$e", t.backend == Nvidia ? "$st.bc" : nothing
		try
			io(() -> write(s, ir), "cannot write native LLVM IR")
			debug("native source key=$k path=$s")
			# The cached artifact hides the compiler's view of the kernel, so residency is checked only here.
			for r in compile_native_artifact(t, s, o, b, k)
				debug("native kernel $(r.name) registers=$(r.registers) scalars=$(r.scalars) occupancy=$(r.occupancy) waves per SIMD")
				require(r.occupancy != 0, "native kernel $(r.name) cannot be resident at any workgroup width")
			end
			io(() -> Base.Filesystem.rename(o, f), "cannot publish native artifact $f")
		finally
			for q in (s, o, b) q === nothing || try rm(q; force=true) catch end end
		end
	end
	r = io(() -> read(f), "cannot read native artifact $f")
	require(!isempty(r), "native artifact $f is empty")
	NativeArtifact(t, m.layout, m.precision, r, f, layout_storage(m))
end

function shape1(n::Node, rows)
	I, O = n.input, n.output
	if n.op == PContraction
		sp = max(Int(integer_argument(n.argument[1], "native contraction kernel")), 1)
		w = cm(I.channels, sp, "native contraction window"); r = cm(rows, O.length, "native contraction output rows")
		d = ((r, O.channels, w), (w, O.channels, r), (cm(rows, I.length, "native contraction input rows"), I.channels,
			cm(O.channels, sp, "native contraction previous terms")), n.parameters)
	elseif n.op == PScan
		r = cm(rows, I.length, "native scan projection rows")
		d = ((r, O.channels, I.channels), (I.channels, O.channels, r), (r, I.channels, O.channels),
			cm(I.channels, O.channels, "native scan projection parameters"))
	else
		return nothing
	end
	e(x, s) = Tile((nw(v, "native $s contraction $c") % UInt32 for (v, c) in zip(x, "MNK"))...)
	NativeContractionShapes(e(d[1], "forward"), e(d[2], "gradient"), e(d[3], "previous"), d[4] % Int)
end
Recipe.native_contraction_shapes(g, rows) = Union{Nothing,NativeContractionShapes}[shape1(n, rows) for n in g.nodes]
function Recipe.native_attention_shared_values(e::Tile, full::Bool)
	m, n, k = W(e.m), W(e.n), W(e.k)
	q = u(m * k, "native attention query tile overflows"); y = u(n * k, "native attention key tile overflows")
	p = u(m * n, "native attention pair tile overflows"); s = "native attention shared values overflow"
	f, a, b, x = u(2q + 2y + 2p + 3m, s), u(3q + 2y + 2p + m, s), u(2q + 4y + 2p + m, s), u(4q + 2m * m + m, s)
	UInt32(max(f, a, b, full ? x : 0))
end
function Recipe.native_attention_tile(l, w, sv, qt)
	require(l != 0 && w != 0 && sv != 0 && qt != 0, "native attention tile inputs are empty")
	l, w, sv = Int(l), Int(w), Int(sv); q = min(l, Int(qt))
	while true
		v = W(q) * w; ff, fk, gf, gk, kf, kk = 2v + 3q, 2(w + q), 3v + q, 2w + 2q, 2v + q, 4w + 2q
		u(max(ff, fk, gf, gk, kf, kk), "native attention tile overflows")
		y = min(max(sv - ff, 0) ÷ fk, max(sv - gf, 0) ÷ gk, max(sv - kf, 0) ÷ kk)
		x = u(4v + 2W(q)^2 + q, "native attention matrix tile overflows")
		y != 0 && (q != l || x <= sv) && return Tile(q, min(y, l), w)
		(q -= 1) == 0 && err("native attention tile does not fit the device")
	end
end
function att1(n::Node, sv, qt)
	n.op != PAttention && return nothing
	h = integer_argument(n.argument[1], "native attention heads") % UInt32
	c = narrow(n.output.channels, "native attention channels") % UInt32; l = narrow(n.output.length, "native attention length") % UInt32
	require(c % h == 0, "native attention head partition is invalid")
	native_attention_tile(l, c ÷ h, sv, qt)
end
Recipe.native_attention_tiles(g, sv, qt) = Union{Nothing,Tile}[att1(n, sv, qt) for n in g.nodes]
Recipe.native_tiles(t, w, r) = ((t, w) = (U(t), U(w)); require(t != 0 && w != 0, "$r is empty"); v = ca(t ÷ w, t % w != 0, r); v > typemax(Int) ? v : Int(v))
"Chunk partials the k lanes of one job exchange, in model-sized elements (state values × ratio) per chunk."
function Recipe.native_contraction_partial_per_chunk(m, n, rm, rn, b, ra)
	m, n, rm, rn, b, ra = Int.((m, n, rm, rn, b, ra))
	o = u(W(max(m ÷ rm, 1)) * max(n ÷ rn, 1), "native contraction lane count overflows")
	r = u(W(rm) * rn, "native contraction register tile overflows"); P = "native contraction partial region overflows"
	s = W(min(o, max(b ÷ 2, 1))) * r; c = W(max(n ÷ rn, 1)) * rn; u(s, P); u(c, P); u(s + c, P)
	UInt32(u((s + c) * ra, P))
end
"Staged operands and the chunk partials share local memory; a tile too wide for one chunk narrows M, then N lanes."
function Recipe.native_contraction_tile(L::Tile, rm, rn, b, sv, fr, ra, mx::Bool)
	rm, rn, b, sv, fr, ra = Int.((rm, rn, b, sv, fr, ra)); Lm, Ln, Lk = Int(L.m), Int(L.n), Int(L.k)
	require(rm != 0 && rn != 0 && b != 0 && fr != 0 && ra != 0, "native contraction tile inputs are empty")
	if mx
		v = b ÷ 32; require(v != 0, "native matrix contraction has no wave")
		m = u(W(v) * 16, "native matrix contraction M tile overflows"); n = max(b ÷ 2, 32)
		o = sv ÷ u(W(m) + n, "native matrix contraction tile width overflows")
		k = min(u(W(cld(Lk, fr)) * fr, "native matrix contraction K tile overflows"), o - o % fr)
		require(k != 0, "native matrix contraction tile does not fit the device")
		return Tile(m, n, k)
	end
	ln = min(cld(Ln, rn), max(isqrt(b), 1)); wm = cld(Lm, rm); lm = min(wm, b ÷ ln)
	while true
		m = u(W(lm) * rm, "native contraction M tile overflows"); n = u(W(ln) * rn, "native contraction N tile overflows")
		sk = sv ÷ u(W(m) + n, "native contraction tile width overflows")
		o = min(sk, u(W(sv ÷ Int(native_contraction_partial_per_chunk(m, n, rm, rn, b, ra))) * fr, "native contraction partial K overflows"))
		# Whole chunks per multi-tile walk; a walk staged in one tile keeps its exact length.
		o >= Lk && return Tile(m, n, Lk)
		o >= fr && return Tile(m, n, o - o % fr)
		if lm > 1
			lm -= 1
		else
			ln > 1 || err("native contraction tile does not fit the device"); ln -= 1; lm = min(wm, b ÷ ln)
		end
	end
end
function Recipe.native_contraction_shared_values(e::Tile, rm, rn, b, fr, ra, mx::Bool)
	t = W(e.m) + e.n; M = "native contraction shared values overflow"; u(t, M); s = u(t * e.k, M)
	mx && return UInt32(s)
	c = cld(Int(e.k), Int(fr)); p = native_contraction_partial_per_chunk(e.m, e.n, rm, rn, b, ra)
	UInt32(max(s, u(W(c) * p, "native contraction partial region overflows")))
end
"Gradient values plus split-K scratch: one row per K partition, rows aligned to NATIVE_SCRATCH_ROW_VALUES."
function Recipe.native_gradient_values(P, cs)
	S = U(NATIVE_SCRATCH_ROW_VALUES); a(x) = (x = U(x); x % S == 0 ? x : x + (S - x % S)); z = U(0)
	for c in cs
		c === nothing && continue
		g, h = c.gradient_shape, c.gradient
		sp = max(min(cld(U(g.k), min(NATIVE_SPLIT_SPAN, NATIVE_MATRIX_SPLIT_SPAN)), NATIVE_K_PARTITIONS), 1)
		j = cm(native_tiles(g.m, h.m, "native split-K M tiles"), native_tiles(g.n, h.n, "native split-K N tiles"), "native split-K jobs")
		nw(cm(j, sp, "native split-K tasks"), "native split-K tasks")
		sp > 1 && (z = max(z, cm(sp, a(c.parameters), "native split-K scratch")))
	end
	v = ca(a(P), z, "native gradient and split-K scratch"); nw(v, "native gradient and split-K scratch"); Int(v)
end
gthreads(g) = (t = W(g.groups) * g.block; t == 0 || t > 0xffffffff ? err("GPU launch size overflows") : UInt32(t))
G(c, b) = (groups=UInt32(c), block=UInt32(b))
function Recipe.geometry(cus, wave, wg, lds, gpc, r)
	require(wave != 0 && wave <= wg && wave <= r.max_block, "GPU wave exceeds kernel workgroup")
	w = min(gpc, wg ÷ wave, r.max_block ÷ wave); require(w != 0, "GPU has no resident wave")
	b = u(W(w) * wave, "GPU workgroup size overflows"); require(r.shared <= lds, "GPU tile exceeds local memory"); G(cus, b)
end
function Recipe.amd_resources(cus, wave, wg, lds, ws, r)
	b = u(W(wave) * ws, "AMD workgroup size overflows")
	require(wave != 0 && b <= wg && b <= r.max_block && ws != 0, "AMD workgroup geometry is invalid")
	require(r.shared <= lds, "AMD workgroup exceeds local memory"); G(cus, b)
end
Recipe.nvidia_resources(cus, wave, wg, bl, sl, wpc, r) = (require(r.shared <= bl, "Nvidia tile exceeds workgroup shared memory"); geometry(cus, wave, wg, sl, wpc, r))
end

# ==== BUNDLE ====
@sec BUNDLE begin
# recipe.rs 5751-6893 (mod bundle): .ogdl save/load/restore, artifact key, run_infer. Budget 32,000. Parity: bundle bytes.
end

# ==== QUANT ====
@sec QUANT begin
# recipe.rs 7244-9031: fp16, k/i-quant codecs, IQ grids (encode tables compactly), StorageFormat. Budget 58,000. Parity: bytes.
end

# ==== API ====
@sec API begin
# recipe.rs 7104-7243, 9032-9277, 18707-18717, 18853-19621 and cli.rs: Model DSL, Recipe, losses, metrics, Train, reports, CLI. Budget 36,000.
# Placeholder until the API section lands: usage check only.
Recipe.julia_main() = (println(stderr, "usage: recipe [--device <name>] <source.jl> [export]"); Cint(2))
end

# ==== GRAPH ====
@sec GRAPH begin
# recipe.rs 9409-10397, 10588-10642: graph storage, compile, lowering, stored_graph. Budget 28,000. Parity: graph dumps.
end

# ==== GPU ====
@sec GPU begin
# recipe.rs 10643-13468: NativeTape, Buffer, Kernel, Cuda, Hsa, Gpu, NativeProgram, devices (no CPU device). Budget 72,000. Needs GPU.
# Spike bindings (recipe-julia-spike, U0): dlopen/dlsym + ccall. Adapt to recipe.rs's Library/Cuda/Hsa structs.
const CU_SYMBOLS = (:cuInit, :cuDriverGetVersion, :cuDeviceGetCount, :cuDeviceGet, :cuDeviceGetName, :cuDeviceGetAttribute,
	:cuDeviceTotalMem_v2, :cuCtxCreate_v2, :cuCtxDestroy_v2, :cuCtxSetCurrent, :cuCtxSynchronize, :cuModuleLoadData,
	:cuModuleUnload, :cuModuleGetFunction, :cuMemAlloc_v2, :cuMemFree_v2, :cuMemcpyHtoD_v2, :cuMemcpyDtoH_v2, :cuLaunchKernel, :cuGetErrorName)
struct Drv; library::Ptr{Cvoid}; f::Dict{Symbol,Ptr{Cvoid}} end
drv_open(lib, syms) = (l = Libdl.dlopen(lib; throw_error=false); l === nothing ? nothing : Drv(l, Dict(s => Libdl.dlsym(l, s) for s in syms)))
cu_open() = drv_open(MR["nvidia-runtime"], CU_SYMBOLS)
function cu_error(d::Drv, s::Cint)
	n = Ref{Ptr{UInt8}}(C_NULL)
	ccall(d.f[:cuGetErrorName], Cint, (Cint, Ref{Ptr{UInt8}}), s, n) == 0 && n[] != C_NULL ? unsafe_string(n[]) : "CUDA error $s"
end
cu_check(d::Drv, s::Cint, a) = s == 0 || err("NVIDIA $a: $(cu_error(d, s))")
struct CuDevice; index::Int; handle::Cint; name::String; target::String; sms::Int; memory::UInt64 end
function cu_devices(d::Drv)
	cu_check(d, ccall(d.f[:cuInit], Cint, (Cuint,), 0), "initialization")
	c = Ref{Cint}(0); cu_check(d, ccall(d.f[:cuDeviceGetCount], Cint, (Ref{Cint},), c), "device enumeration")
	map(0:c[]-1) do i
		h = Ref{Cint}(0); cu_check(d, ccall(d.f[:cuDeviceGet], Cint, (Ref{Cint}, Cint), h, i), "device query")
		nm = zeros(UInt8, 256); cu_check(d, ccall(d.f[:cuDeviceGetName], Cint, (Ptr{UInt8}, Cint, Cint), nm, 256, h[]), "name query")
		at(k) = (o = Ref{Cint}(0); cu_check(d, ccall(d.f[:cuDeviceGetAttribute], Cint, (Ref{Cint}, Cint, Cint), o, k, h[]), "attribute $k"); o[])
		m = Ref{Csize_t}(0); cu_check(d, ccall(d.f[:cuDeviceTotalMem_v2], Cint, (Ref{Csize_t}, Cint), m, h[]), "VRAM size")
		CuDevice(i, h[], unsafe_string(pointer(nm)), "sm_$(at(Cint(75)))$(at(Cint(76)))", at(Cint(16)), m[])  # compute major/minor, SM count
	end
end
const HS_SYMBOLS = (:hsa_init, :hsa_shut_down, :hsa_status_string, :hsa_iterate_agents, :hsa_agent_get_info,
	:hsa_amd_agent_iterate_memory_pools, :hsa_amd_memory_pool_get_info, :hsa_amd_memory_pool_allocate, :hsa_amd_memory_pool_free,
	:hsa_amd_agents_allow_access, :hsa_memory_copy, :hsa_queue_create, :hsa_queue_destroy, :hsa_queue_add_write_index_scacq_screl,
	:hsa_signal_create, :hsa_signal_destroy, :hsa_signal_store_screlease, :hsa_signal_wait_scacquire,
	:hsa_code_object_reader_create_from_memory, :hsa_code_object_reader_destroy, :hsa_executable_create_alt,
	:hsa_executable_load_agent_code_object, :hsa_executable_freeze, :hsa_executable_get_symbol_by_name,
	:hsa_executable_symbol_get_info, :hsa_executable_destroy)
# hsa.h values: agent info NAME 0, WAVEFRONT_SIZE 6, DEVICE 17, AMD CU count 0xA002, product 0xA009; pool SEGMENT 0, GLOBAL_FLAGS 1,
# flags KERNARG_INIT 1, COARSE_GRAINED 4; symbol KERNEL_OBJECT 22, KERNARG_SIZE 11, GROUP_SIZE 13, PRIVATE_SIZE 14;
# packet KERNEL_DISPATCH 2, FENCE_SYSTEM 2, SIGNAL_LT 2, WAIT_BLOCKED 0. hsa_queue_t: base_address @8, doorbell @16, size @24.
hs_open() = drv_open(MR["hsa-runtime"], HS_SYMBOLS)
function hs_check(d::Drv, s::Cint, a)
	s == 0 && return
	t = Ref{Ptr{UInt8}}(C_NULL); ccall(d.f[:hsa_status_string], Cint, (Cint, Ref{Ptr{UInt8}}), s, t)
	err("AMD $a: " * (t[] == C_NULL ? "HSA status $s" : unsafe_string(t[])))
end
mutable struct Collected; items::Vector{UInt64} end
hs_collect(h::UInt64, p::Ptr{Cvoid})::Cint = (push!((unsafe_pointer_to_objref(p)::Collected).items, h); Cint(0))
hs_iterate(f) = (c = Collected(UInt64[]); GC.@preserve c f(@cfunction(hs_collect, Cint, (UInt64, Ptr{Cvoid})), pointer_from_objref(c)); c.items)
hs_agents(d) = hs_iterate((cb, p) -> hs_check(d, ccall(d.f[:hsa_iterate_agents], Cint, (Ptr{Cvoid}, Ptr{Cvoid}), cb, p), "agent iteration"))
hs_pools(d, a::UInt64) = hs_iterate((cb, p) -> hs_check(d, ccall(d.f[:hsa_amd_agent_iterate_memory_pools], Cint, (UInt64, Ptr{Cvoid}, Ptr{Cvoid}), a, cb, p), "pool iteration"))
hs_info(d, a::UInt64, k, ::Type{T}) where {T} = (o = Ref{T}(); hs_check(d, ccall(d.f[:hsa_agent_get_info], Cint, (UInt64, Cint, Ptr{Cvoid}), a, k, o), "agent info $k"); o[])
hs_info_string(d, a::UInt64, k) = (b = zeros(UInt8, 64); hs_check(d, ccall(d.f[:hsa_agent_get_info], Cint, (UInt64, Cint, Ptr{UInt8}), a, k, b), "agent info $k"); unsafe_string(pointer(b)))
function hs_pool(d, agent::UInt64, mask::UInt32)
	for p in hs_pools(d, agent)
		s, f = Ref{UInt32}(99), Ref{UInt32}(0)
		hs_check(d, ccall(d.f[:hsa_amd_memory_pool_get_info], Cint, (UInt64, Cint, Ptr{Cvoid}), p, 0, s), "pool segment")
		s[] == 0 || continue
		hs_check(d, ccall(d.f[:hsa_amd_memory_pool_get_info], Cint, (UInt64, Cint, Ptr{Cvoid}), p, 1, f), "pool flags")
		f[] & mask != 0 && return p
	end
	err("AMD pool with flags $mask is absent")
end
struct HsDevice; index::Int; agent::UInt64; target::String; name::String; cus::Int; wave::Int end
"GPU agents (device type 1), discrete or not."
hs_devices(d) = (r = HsDevice[]; for a in hs_agents(d)
	hs_info(d, a, 17, UInt32) == 1 && push!(r, HsDevice(length(r), a, hs_info_string(d, a, 0), hs_info_string(d, a, 0xA009), Int(hs_info(d, a, 0xA002, UInt32)), Int(hs_info(d, a, 6, UInt32))))
end; r)
hs_symbol_info(d, s::UInt64, k, ::Type{T}) where {T} = (o = Ref{T}(zero(T)); hs_check(d, ccall(d.f[:hsa_executable_symbol_get_info], Cint, (UInt64, Cint, Ptr{Cvoid}), s, k, o), "symbol info $k"); o[])
function hs_allocate(d, pool::UInt64, n::Integer, agent::UInt64)
	o = Ref{Ptr{Cvoid}}(C_NULL)
	hs_check(d, ccall(d.f[:hsa_amd_memory_pool_allocate], Cint, (UInt64, Csize_t, UInt32, Ref{Ptr{Cvoid}}), pool, n, 0, o), "allocation")
	hs_check(d, ccall(d.f[:hsa_amd_agents_allow_access], Cint, (UInt32, Ref{UInt64}, Ptr{UInt32}, Ptr{Cvoid}), 1, Ref(agent), C_NULL, o[]), "allow access")
	o[]
end
hs_free(d, p::Ptr{Cvoid}) = hs_check(d, ccall(d.f[:hsa_amd_memory_pool_free], Cint, (Ptr{Cvoid},), p), "free")
hs_copy(d, dst::Ptr, src::Ptr, n) = hs_check(d, ccall(d.f[:hsa_memory_copy], Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, src, n), "copy")
end

# ==== ML ====
@sec ML begin
# recipe.rs 13469-14871: surrogate, predictor programs, trees, SVM, forest, Bayes, XGBoost, LightGBM, CatBoost, k-means, kNN. Budget 36,000. Parity: predictions.
end

# ==== DATAA ====
@sec DATAA begin
# recipe.rs 15440-17124: Data API, directory samples, sample tables, JPEG/PNG, prepare. Budget 45,000. Parity: prepared samples, schema.
end

# ==== DATAB ====
@sec DATAB begin
# recipe.rs 17125-18706: table decoders (SQLite, inflate, HDF5, zip, npy, JSON, XML, CSV), features, shuffle. Budget 58,000. Parity: tables.
end

end # module
