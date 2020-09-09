using AutoLM
using Test
include("custom_data_loading.jl")

image_data = load_imgs("C:/Users/imidi/.julia/dev/AutoLM.jl/test/test_data", (128,128,128))
landmark_data = read_landmarks("C:/Users/imidi/.julia/dev/AutoLM.jl/test/test_data", 10, "@1")

@testset "AutoLM.jl" begin
    aligned, aligned_lm, reco = align_principal(image_data, landmark_data, 192)
end
