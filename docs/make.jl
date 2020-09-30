push!(LOAD_PATH, "C:\\Users\\immanueldiener\\.julia\\dev")
using Documenter
using AutoLandmarking

makedocs(sitename="AutoLandmarking.jl")

deploydocs(
    repo = "github.com/imi-diener/AutoLandmarking.jl.git",
)
