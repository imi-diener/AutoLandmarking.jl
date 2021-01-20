aligned, aligned_lm, reco = align_principal(images, lms_on_surface, 192)

resized, resized_lm, scales = resize_relevant(aligned, aligned_lm, 128)
aligned = nothing

images1, imnames1 = load_imgs("C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/bonobo_volumes_large", (256,256,256), true)
images2, imnames2 = load_imgs("C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/chimp_volumes_large", (256,256,256), true)
images = cat(images1, images2, dims=4)
images = Float32.(images)
images1 = nothing
images2 = nothing

landmark_data = read_landmarks("C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/bonobo_volumes_large", 22, "@1")
landmark_data2 = read_landmarks("C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/chimp_volumes_large", 22, "@1")
landmarks = cat(landmark_data, landmark_data2, dims=2)./10


X_train, X_test, y_train, y_test = regular_train_test_split_2d(images, landmarks)


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
ImageView.imshow(X_test)


complete1 = depth_map_all_sides(X_train)
X_train=nothing
complete2 = depth_map_all_sides(flip1)
flip1=nothing
complete3 = depth_map_all_sides(flip2)
flip2=nothing
complete4 = depth_map_all_sides(flip3)
flip3=nothing
complete5 = depth_map_all_sides(rot)
rot=nothing
complete6 = depth_map_all_sides(rot2)
rot2=nothing
complete7 = depth_map_all_sides(rot3)
rot3=nothing
complete8 = depth_map_all_sides(jit)
jit=nothing
complete9 = depth_map_all_sides(jit2)
jit2=nothing
complete10 = depth_map_all_sides(jit3)
jit3=nothing
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
for i in 1:150
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
  if sum(acc2)[1]<23.0
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
Plots.png(plt, "C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/mandibles_training.png")


# Data analysis mandibles ==================
response = cpu(predict_set(gpu(X_test), model))
X_train2, X_test2, y_train2, y_test2 = regular_train_test_split_2d(images, landmarks)
on_vol = AutoLandmarking.landmark_to_surface(X_test2, response, 6)
on_vol_no4 = on_vol[:, cat(1:3,5:19,dims=1)]
response_no4 = response[:, cat(1:3,5:19,dims=1)]

function print_costs(y, resp)
  costs=[]
  for i in 1:size(resp,2)
    dev = sum((y[:,i] .- resp[:,i]) .^2)
    push!(costs, dev)
    println(i, "    ", dev)
  end
  return costs
end

costs_resp = print_costs(y_test_no4, response_no4)

# puting on volume makes it worse

three_d_train = AutoLandmarking.to_3d_array(y_train2)
three_d_test = AutoLandmarking.to_3d_array(response_no4)

aligned_train = AutoLandmarking.align_all(three_d_train)
aligned_test = AutoLandmarking.align_all(three_d_test)

mean_train, stdevs = mean_shape(aligned_train)

distances = proc_distance(mean_train, aligned_test, stdevs)
dists = []
for i in 1:18
  push!(dists, distances[i])
end

for i in 1:18
  println("ind $i cost is ", costs_resp[i], "  and dist is ", dists[i])
end

uncertainties = []
for i in 1:18
  means, stddev = AutoLandmarking.response_distribution(model, gpu(X_test_no4[:,:,:,i:i]), gpu(y_test_no4[:,i:i]), 100)
  println("$i has uncevertaty", sum(stddev), "  and cost  ", costs_resp[i])
  push!(uncertainties, sum(stddev))
end
# 6 are over 40

dists_conf = dists[findall(x->x<4,dists)]
resp_conf = response_no4[:,findall(x->x<4,dists)]
volumes_conf = X_test2[:,:,:,findall(x->x<4,dists)]
y_test_conf = y_test_no4[:,findall(x->x<4,dists)]
X_test_conf = X_test_no4[:,:,:,findall(x->x<4,dists)]
AutoLandmarking.change_values!(y_test_conf, 26.0, 0, >)
ImageView.imshow(X_test_conf)

is_bonobo = zeros(97,1)
is_bonobo[1:37,1] .= 1
imnames = cat(imnames1, imnames2, dims=1)
test_names = []
train_names = []
is_bonobo_train = []
is_bonobo_test = []
for i in 1:97
  if i % 5 == 0
    push!(test_names, imnames[i])
    push!(is_bonobo_test, is_bonobo[i])
  else
    push!(train_names, imnames[i])
    push!(is_bonobo_train, is_bonobo[i])
  end
end

is_bonobo_test_no4 = is_bonobo_test[cat(1:3,5:19,dims=1)]
is_bonobo_conf = is_bonobo_test_no4[findall(x->x<41,dists)]
test_names = test_names[cat(1:3,5:19,dims=1)]
names_conf = test_names[findall(x->x<41,dists)]
AutoLandmarking.array_to_lm_file("C:/Users/immanueldiener/Desktop/Master/worst_pred.landmarkAscii", response_no4[:,18])

import ImageView
using DelimitedFiles
using Statistics

y_test_no4 = y_test[:,cat(1:3,5:19,dims=1)]
X_test_no4 = X_test[:,:,:,cat(1:3,5:19,dims=1)]
ImageView.imshow(X_test_no4)


ImageView.imshow(depth_map_all_sides(X_test_no4))
test_names[14]
all_devs = zeros(14, 4)
devs_per_point = zeros(14*22, 2)

AutoLandmarking.change_values!(y_test_no4, 26.0, 0.0, >)

testmode!(model)
for i in 1:14
  acc2, max1, min1 = AutoLandmarking.avg_accuracy_per_point(model, gpu(X_test_conf[:,:,:,i:i]), gpu(y_test_conf[:,i:i]), 3)
  all_devs[i,1] = sum(acc2)
  all_devs[i,2] = dists_conf[i]
  all_devs[i,3] = i
  if i in findall(x->x<41,dists)
    all_devs[i,4] = 1
  end
end
minimum(all_devs[:,1])
all_devs
using Distances
for i in 1:14
  for j in 1:22
    devs_per_point[(i-1)*22+j, 1] = euclidean(resp_conf[j*3-2:j*3, i], y_test_conf[j*3-2:j*3, i])
    devs_per_point[(i-1)*22+j, 2] = j
    if j == 2
      println(i, "   ", euclidean(resp_conf[j*3-2:j*3, i], y_test_conf[j*3-2:j*3, i]))
    end
  end
end

devs_per_point

AutoLandmarking.change_values!(y_test, 25.8, 25.6, >)
test_names[18]

out = hcat(all_devs, names_conf)

writedlm("C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/devs_per_ind.txt", out)
writedlm("C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/devs_per_point.txt", devs_per_point)


# timing GPU : 12000/min, CPU: 110/min
@time begin
  predict_set(X_train[:,:,:,1:50], model_cpu)
end



# export of coordinates
out_test = hcat(names_conf, is_bonobo_conf, resp_conf')
out_train = hcat(train_names, is_bonobo_train, y_train')
out_all = vcat(out_train, out_test)
is_train = zeros(92, 1)
is_train[1:78] .= 1
output_all = hcat(out_all[:,1:2], is_train, out_all[:,3:end])

writedlm("C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/coords_pred.txt", out_test)
writedlm("C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/coords_train.txt", out_train)
writedlm("C:/Users/immanueldiener/Desktop/Master/master_data/mandibles/coords_all.txt", output_all)
# ===========================================

using BSON
testmode!(model)
model_cpu = cpu(model)
BSON.@save "C:/Users/immanueldiener/Desktop/Master/master_data/model_mandibles.bson" model_cpu
BSON.@load "C:/Users/immanueldiener/Desktop/Master/master_data/model_mandibles.bson" model_cpu
model = gpu(model_cpu)
