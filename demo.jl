using AutoLM
using Flux
using CuArrays
import ImageView

push!(LOAD_PATH, "C:/Users/immanueldiener/.julia/dev")

# Import data with custom import functions
image_data, imname = load_imgs("C:/Users/immanueldiener/Desktop/Master/scripts/no_enamel", (128,128,128), false)
landmark_data = read_landmarks("C:/Users/immanueldiener/Desktop/Master/scripts/no_enamel", 10, "@1")

# X and Y coordinates are swapped in avizo files, wo I'll swap them back here.
#this step is essential for all the data augmentation to work.
landmark_data = swap_xy(landmark_data)

#some images have a higher value in the fill voxels than in the actual object voxels. This will fix the issue.
# The highest value must always be the filler value!
image_data[:,:,:,[109,130]] = image_data[:,:,:,[109,130]] .- 1
image_data[:,:,:,[109,130]] = image_data[:,:,:,[109,130]] .* -1

image_data2, imname2 = load_imgs("C:/Users/immanueldiener/Desktop/Master/master_data\\additional_teeth_volumes\\", (128,128,128), true)
landmark_data2 = read_landmarks("C:/Users/immanueldiener/Desktop/Master/master_data\\additional_teeth_volumes\\", 10, "@1")

#Swap X and Y again
landmark_data2 = swap_xy(landmark_data2)

#all these images have reversed fill and object values. Tha highest value must always be the filler value!
image_data2 = (image_data2 .- 1) .* -1

images = cat(image_data, image_data2, dims=4)
lms = cat(landmark_data, landmark_data2, dims=2)

#put all the landmarks that are not yet on the surface of an object onto the landmark
lms_on_surface = landmark_to_surface(images, lms, 10)


#define training and testing sets
X_train = images[:,:,:,setdiff(1:243,86:125)]
X_test = images[:,:,:,86:125]
y_train = lms_on_surface[:, setdiff(1:243,86:125)]
y_test = lms_on_surface[:, 86:125]

#augment the training data for better learning and less overfitting
X_train, y_train = mirror_vol(X_train, y_train)
flip1, lm_flipped1 = flip_volume_front(X_train, y_train)
flip2, lm_flipped2 = flip_volume_side(X_train, y_train)
flip3, lm_flipped3 = flip_volume_front(flip1, lm_flipped1)
flip4, lm_flipped4 = flip_volume_side(flip3, lm_flipped3)

# we'll be using flip2 and flip4 and rotate them by 30/60 and 20/70 degrees respectively
flip2_rot1, lm_flip2_rot1 = rotate_volumes(flip2, lm_flipped2, 30)
flip2_rot2, lm_flip2_rot2 = rotate_volumes(flip2, lm_flipped2, 60)

flip4_rot1, lm_flip4_rot1 = rotate_volumes(flip4, lm_flipped4, 20)
flip4_rot2, lm_flip4_rot2 = rotate_volumes(flip4, lm_flipped4, 70)

# lets also rotate the original data (including the mirror images)
og_rot1, lm_og_rot1 = rotate_volumes(X_train, y_train, 50)
og_rot2, lm_og_rot2 = rotate_volumes(X_train, y_train, 80)

# jittering around of the object inside the volume. Jittering is landmark based and
# a 10 Voxel padding will be added after the minimum/maximum landmark coordinate in
# any dimension.

jit1, lm_jit1 = jitter_3D(flip2_rot1, lm_flip2_rot1, 10)
jit2, lm_jit2 = jitter_3D(flip4_rot2, lm_flip4_rot2, 10)
jit3, lm_jit3 = jitter_3D(og_rot2, lm_og_rot2, 10)

#lets also use aligned versions of the original dataset
aligned, lm_aligend, reco = align_principal(X_train, y_train, 192)
resized, lm_resized = resize_relevant(aligned, lm_aligend, 128)
aligned = nothing

jit4, lm_jit4 = jitter_3D(X_train, y_train, 10)

#generate all the depthmaps for input into the network
complete1 = depth_map_all_sides(flip2_rot1)
complete2 = depth_map_all_sides(flip2_rot2)
complete3 = depth_map_all_sides(flip4_rot1)
complete4 = depth_map_all_sides(flip4_rot2)
complete5 = depth_map_all_sides(og_rot1)
complete6 = depth_map_all_sides(og_rot2)
complete7 = depth_map_all_sides(jit1)
complete8 = depth_map_all_sides(jit2)
complete9 = depth_map_all_sides(jit3)
complete10 = depth_map_all_sides(resized)
complete11 = depth_map_all_sides(flip1)
complete12 = depth_map_all_sides(flip3)
complete13 = depth_map_all_sides(jit4)

X_train = cat(complete1, complete2, complete3, complete4, complete5, complete6, complete7, complete8,
    complete9, complete10, complete11, complete12, complete13, dims=4)
y_train = cat(lm_flip2_rot1, lm_flip2_rot2, lm_flip4_rot1, lm_flip4_rot2, lm_og_rot1, lm_og_rot2,
    lm_jit1, lm_jit2, lm_jit3, lm_resized, lm_flipped1, lm_flipped3, lm_jit4, dims=2)

# make depthmaps of the testing data
X_test = depth_map_all_sides(X_test)

# define the cost function
cost(x, y) = sum((model(x)-y).^2)|>gpu

#define the model
model = Flux.mapleaves(cu, AutoLM.vgg19)

# define the trainingrate and optimiser
opt = Flux.ADAM(0.000015)

# redefinition of the dropout function to work with testmode!() since this
# functionality is not working at the moment. This step only needs to be done
# if you wish to perform uncertainty estimation using the dropout method (AutoLM.response_distribution())
# after training.
using Random
function Flux.dropout(x, p; dims = :)
    q = 1 - p
    y = rand!(similar(x, Flux._dropout_shape(x, dims)))
    y .= Flux._dropout_kernel.(y, p, q)
    x .* y
end

import Zygote
Zygote.@adjoint function Flux.dropout(x, p; dims = :)
   q = 1 - p
   y = rand!(similar(x, Flux._dropout_shape(x, dims)))
   y .= Flux._dropout_kernel.(y, p, q)
   return x .* y, Δ -> (Δ .* y, nothing)
end
Zygote.refresh()

function run_model(modell, X, y)
  train_data = Flux.mapleaves(cu, X)
  train_labels = Flux.mapleaves(cu, y)
  dataset = Flux.Data.DataLoader(train_data, train_labels, batchsize = 4, shuffle=true)
  Flux.train!(cost, params(model), dataset, opt)
  testmode!(model)
  cosima = cost_whole_data_2D(train_data, train_labels, cost)
  testmode!(model, false)
  return cosima
end


# create lists to store metrics
accs = []
costs = []

# run the training for 300 epochs. batches of 128 will be loaded onto the GPU,
# which will be further subdevided into minibatches of 4 (as defined in AutoML.run_model).
for i in 1:300
  costr = 0
  for j in 1:128:size(X_train, 4)
    if j+127 > size(X_train, 4)
      train_data = X_train[:,:,:,j:end]
      train_labels = y_train[:,j:end]
    else
      train_data = X_train[:,:,:,j:j+127]
      train_labels = y_train[:,j:j+127]
    end
    cosima = run_model(model, train_data, train_labels)
    costr = costr + cosima
    train_data = nothing
    train_labels = nothing
    # GC.gc()
  end
  push!(costs, costr)
  println("epoch ", i, " finished")
  println(costr)
  testmode!(model)
  acc2, max1, min1 = AutoLM.avg_accuracy_per_point(model, gpu(X_test[:,:,:,:]), gpu(y_test[:,:]), 3)
  testmode!(model, false)
  println("median deviation per point on testing dataset: ", acc2, "with maximum", max1)
  println("median sum is ", sum(acc2))
  push!(accs, sum(acc2))
  if sum(acc2)[1]<7.5
    break
  end
end
