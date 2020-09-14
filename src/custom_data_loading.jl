import Images
"""
    load_imgs(path, dims)

load the image data from the prepared data. This is a custom function
designed to read data stored in this specific fashion
"""
function load_imgs(path, dims, numerical)
    inds = 0
    for (root, dirs, files) in walkdir(path)
        if length(dirs) != 0
            inds = length(dirs)
        end
    end
    image_matrix = zeros(dims[1], dims[2], dims[3], inds)
    vol_names = []
    for (root, dirs, files) in walkdir(path)
        for ind in 1:length(dirs)
          push!(vol_names, dirs[ind])
            ind_path = joinpath(path, dirs[ind])
            items = readdir(ind_path)
            if numerical
                sort!(items, by=x->parse(Int,split(x, ".")[1]))
            end
            for i in items
                if i[end-3:end] != ".tif"
                    deleteat!(items, findall(x->x==i, items))
                end
            end
            for img in 1:length(items)
                im_path = joinpath(ind_path, items[img])
                image = Images.load(im_path)
                image = Images.Gray.(image)
                mat = convert(Array{Float32}, image)
                if size(mat, 1) != 128
                end
                image_matrix[:, :, img, ind] = mat
            end
        end
    end
    return image_matrix, vol_names
end

"""
    read_landmarks(path, num_landmarks, group)

General function to read avizo landmark data into an array
"""
function read_landmarks(path, num_landmarks, group)
    whole_dir = readdir(path)
    lm_files = []
    for item in whole_dir
        if '.' in item
            if item[end-4:end] == "Ascii"
                push!(lm_files, joinpath(path, item))
            end
        else
            continue
        end
    end
    coordinates = zeros(length(lm_files), num_landmarks * 3)
    for file in 1:length(lm_files)
        f = open(lm_files[file])
        fil = readlines(f)
        close(f)
        coords = []
        counter = 0
        for i in fil
            if i == group
                counter = 1
                continue
            end
            if 1<=counter<=num_landmarks
                push!(coords, i)
                counter+=1
            end
        end
        for point in 1:length(coords)
            single_co = split(coords[point], " ")
            single_co = single_co[1:3]
            for co in 1:3
                coordinates[file, point*3 - (3-co)] = parse(Float64, single_co[co][1:end-5]) * 10.0^parse(Float64, single_co[co][end-3:end])
            end
        end
    end
    return coordinates'
end
