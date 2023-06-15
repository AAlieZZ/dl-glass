use tch::{Tensor, kind};
use ndarray::prelude::*;

pub fn image_to_tensor(data:Vec<u8>, dim1:usize, dim2:usize, dim3:usize)-> Tensor{
    // 将Vec转换为三维数组并将颜色值进行归一化处理 
    println!("尝试将Vec转换为{}×{}×{}的三维数组并将颜色值进行归一化处理", dim1, dim2, dim3);
    let inp_data: Array3<f32> = Array3::from_shape_vec((dim1, dim2, dim3), data)
        .expect("Error converting data to 3D array")
        .map(|x| *x as f32/256.0);
    // Array3::from_shape_vec 用于从一个一维的 Vec 创建一个三维数组。
    // 它接受两个参数：一个表示数组形状的元组和一个一维的 Vec。
    // map 用于对数组中的每个元素应用一个函数。
    // 它接受一个闭包作为参数，并对数组中的每个元素调用该闭包，将闭包的返回值作为新数组中对应元素的值。

    // 转成Tensor
    println!("尝试将Tensor转换成 [dim1, dim2*dim3] 结构的张量");
    let inp_tensor = Tensor::from_slice(&inp_data.as_slice().unwrap());
    // 将Tensor转换成 [dim1, dim2*dim3] 结构的张量
    let ax1 = dim1 as i64;
    let ax2 = (dim2 as i64)*(dim3 as i64);
    let shape: Vec<i64>  = vec![ ax1, ax2 ];
    println!("尝试将图像数据重新整形为二维张量");
    let output_data = inp_tensor.reshape(&shape);
    println!("Output image tensor size {:?}", &shape);
    // 许多模型都要求输入数据是二维的，其中第一维表示样本数量，第二维表示特征数量。
    // 因此，将图像数据重新整形为二维张量可以让模型更容易处理。

    output_data
}

pub fn labels_to_tensor(data:Vec<u8>, dim1:usize, dim2:usize)-> Tensor{
    let inp_data: Array2<i64> = Array2::from_shape_vec((dim1, dim2), data)
        .expect("Error converting data to 2D array")
        .map(|x| *x as i64);

    let output_data = Tensor::from_slice(&inp_data.as_slice().unwrap());
    println!("Output label tensor size {:?}", &output_data.size());

    output_data
}

pub fn generate_random_index(array_size: i64, batch_size: i64)-> Tensor{
    let random_idxs = Tensor::randint(array_size, &[batch_size], kind::INT64_CPU);
    random_idxs
}