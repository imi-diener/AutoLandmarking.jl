aligned, aligned_lm, reco = align_principal(images, lms_on_surface, 192)

resized, resized_lm, scales = resize_relevant(aligned, aligned_lm, 128)
aligned = nothing

X_train, X_test, y_train, y_test = regular_train_test_split_3d(resized, resized_lm)


X_train, y_train = (mirror_vol(X_train, y_train))

X_train = Float32.(X_train)

flip1, lm_flip1 = flip_3D(X_train, y_train)
flip2, lm_flip2 = flip_volume_front(flip1, lm_flip1)
flip3, lm_flip3 = flip_volume_side(flip2, lm_flip2)

rot, lm_rot = rotate_volumes(flip2, lm_flip2, 10)
rot2, lm_rot2 = rotate_volumes(X_train, y_train, 20)
rot3, lm_rot3 = rotate_volumes(flip3, lm_flip3, -15)

jit, lm_jit = jitter_3D(flip1, lm_flip1, 10)
jit2, lm_jit2 = jitter_3D(X_train, y_train, 10)
jit3, lm_jit3 = jitter_3D(flip3, lm_flip3, 10)
jit4, lm_jit4 = jitter_3D(rot2, lm_rot2, 10)

using ImageView
ImageView.imshow(X_train)


complete1 = depth_map_all_sides(X_train)
complete2 = depth_map_all_sides(flip1)
complete3 = depth_map_all_sides(flip2)
complete4 = depth_map_all_sides(flip3)
complete5 = depth_map_all_sides(rot)
complete6 = depth_map_all_sides(rot2)
complete7 = depth_map_all_sides(rot3)
complete8 = depth_map_all_sides(jit)
complete9 = depth_map_all_sides(jit2)
complete10 = depth_map_all_sides(jit3)
complete11 = depth_map_all_sides(jit4)

X_train = cat(complete1, complete2, complete3, complete4, complete5, complete6,
    complete7, complete8, complete9, complete10, complete11, dims=4)
y_train = cat(y_train, lm_flip1, lm_flip2, lm_flip3, lm_rot, lm_rot2, lm_rot3, lm_jit, lm_jit2, lm_jit3, lm_jit4, dims=2)

X_train = image_gradients(X_train)
# make depthmaps of the testing data
X_test = depth_map_all_sides(X_test)
X_test = image_gradients(X_test)

# define the cost function
cost(x, y) = sum((model(x)-y).^2)|>gpu

#define the model
model = Flux.mapleaves(cu, AutoLandmarking.vgg19)

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
  acc2, max1, min1 = AutoLandmarking.avg_accuracy_per_point(model, gpu(X_test[:,:,:,:]), gpu(y_test[:,:]), 3)
  testmode!(model, false)
  println("median deviation per point on testing dataset: ", acc2, "with maximum", max1)
  println("median sum is ", sum(acc2))
  push!(accs, sum(acc2))
  if sum(acc2)[1]<7.0
    break
  end
end


#solve issue with arrays inside vector
aks = []
for i in accs
  push!(aks, i[1])
end

# check what the best accuracy is
minimum(aks)

#plot the development of cost and accuracy
import Plots
Plots.plot(aks, legend=:topright, label="sum of deviations per point", color= :red, xlabel="epochs", ylabel = "acuracy")
plt = Plots.plot!(Plots.twinx(), costs, label="training loss", legend=:topleft, ylabel = "loss")
