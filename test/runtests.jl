using AutoLandmarking
using Test
using JLD

@testset "AutoLM.jl" begin
    image_data, imnames = load_imgs(string(Base.@__DIR__, "/test_data"), (128,128,128), false)
    landmark_data = read_landmarks(string(Base.@__DIR__, "/test_data"), 10, "@1")
    solution_load_imgs = JLD.load(string(Base.@__DIR__, "/test_solution/solution_load_imgs.jld"))["solution"]
    solution_read_landmarks = JLD.load(string(Base.@__DIR__, "/test_solution/solution_read_landmarks.jld"))["solution"]
    @test solution_load_imgs == image_data
    @test solution_read_landmarks == landmark_data

    aligned, aligned_lm, reco = align_principal(image_data, landmark_data, 192)
    resized, lm_resized = resize_relevant(aligned, aligned_lm, 128)
    lm_om_volume = landmark_to_surface(resized, lm_resized, 10)
    solution_landmark_to_surface = JLD.load(string(Base.@__DIR__, "/test_solution/solution_landmark_to_surface.jld"))["solution"]
    @test solution_landmark_to_surface == lm_om_volume

    mirrored, lm_mirrored = mirror_vol(resized, lm_om_volume)
    solution_mirror_vol = JLD.load(string(Base.@__DIR__, "/test_solution/solution_mirror_vol.jld"))["solution"]
    @test solution_mirror_vol == mirrored

    flipped, lm_flipped = flip_volume_side(mirrored, lm_mirrored)
    solution_flip_volume_side = JLD.load(string(Base.@__DIR__, "/test_solution/solution_flip_volume_side.jld"))["solution"]
    @test solution_flip_volume_side == flipped

    tilted, lm_tilted = rotate_volumes(flipped, lm_flipped, 30)
    solution_rotate_volumes = JLD.load(string(Base.@__DIR__, "/test_solution/solution_rotate_volumes.jld"))["solution"]
    solution_lm_rotate_volumes = JLD.load(string(Base.@__DIR__, "/test_solution/solution_lms_rotate_volumes.jld"))["solution"]
    @test solution_lm_rotate_volumes == lm_tilted
    @test solution_rotate_volumes == tilted

    depthmaps = depth_map_all_sides(tilted)
    solution_depthmaps = JLD.load(string(Base.@__DIR__, "/test_solution/solution_depthmaps.jld"))["solution"]
    @test solution_depthmaps == depthmaps

end
