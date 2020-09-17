using AutoLM
using Test
using JLD

@testset "AutoLM.jl" begin
    image_data, imnames = load_imgs("C:/Users/immanueldiener/.julia/dev/AutoLM/test/test_data", (128,128,128), false)
    landmark_data = read_landmarks("C:/Users/immanueldiener/.julia/dev/AutoLM/test/test_data", 10, "@1")
    aligned, aligned_lm, reco = align_principal(image_data, landmark_data, 192)
    resized, lm_resized = resize_relevant(aligned, aligned_lm, 128)
    lm_om_volume = landmark_to_surface(resized, lm_resized, 10)
    mirrored, lm_mirrored = mirror_vol(resized, lm_om_volume)
    flipped, lm_flipped = flip_volume_side(mirrored, lm_mirrored)
    tilted, lm_tilted = rotate_volumes(flipped, lm_flipped, 30)
    depthmaps = depth_map_all_sides(tilted)
    solution = JLD.load("C:/Users/immanueldiener/.julia/dev/AutoLM/test/test_solution/solution.jld")["solution"]
    @test solution == depthmaps
end


push!(LOAD_PATH, "C:/Users/immanueldiener/.julia/dev")
