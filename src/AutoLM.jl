module AutoLM

include("data_augmentation.jl")
include("data_preparation.jl")
include("handling_and_utility.jl")
include("models.jl")
include("outliers.jl")
include("custom_data_loading.jl")
# Write your package code here.
export
    # data loading
    load_imgs,
    read_landmarks,

    #data augmentation
    flip_volume_side,
    flip_volume_front,
    mirror_vol,
    flip_2D,
    flip_3D,
    jitter_3D,
    rotate_images,
    rotate_volumes,

    #utility and data preparation
    swap_xy,
    train_test_split_3d,
    regular_train_test_split_3d,
    regular_train_test_split_2d,
    landmark_to_surface,
    image_gradients,
    align_principal,
    resize_relevant,
    translate_lms_back,
    choose_dims,
    depth_map_all_sides,

    #evaluation and training
    cost_whole_data_2D,
    cost_whole_data_3D,
    run_model,
    avg_accuracy_per_point,
    predict_single,
    predict_set,

    #confidence testing and exporting
    save_vols_to_folder,
    response_distribution,
    procrustes_distance_list,
    writedlm

end
