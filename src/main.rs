mod tensor;
mod annotation;
mod image;
mod convnn;

use tensor::{image_to_tensor, labels_to_tensor, generate_random_index};
use crate::image::{Image, RESIZE};
use annotation::Annotation;
use convnn::ConvNN;
use tch::{nn, nn::{ModuleT, OptimizerConfig}, Device, Kind, Tensor};

const N_EPOCHS: i64 = 10; // 迭代次数
const TRAIN_SIZE: usize = 6760;
const ALL_SIZE: usize = 8436;
const BATCH_SIZE: i64 = 128;
const XML_PATH: &str = "./GlassCoverDefectDataset/GlassCover_datset/Annotations/";

fn data_and_lbl(index: Vec<i64>) -> (Image, Vec<u8>) {
    println!("尝试打开{}",format!("{}{}", XML_PATH, format!("{:0>6}", index[0]) + ".xml"));
    let mut annotation = Annotation::from_file(format!("{}{}", XML_PATH, format!("{:0>6}", index[0]) + ".xml")).expect("读取标注文件错误");
    let mut data = Image::new(annotation.get_path()).expect("读取图像错误");

    let mut lbl: Vec<u8> = Vec::new();
    match annotation.get_name().get(1) {
        None => lbl.push(annotation.get_name()[0]),
        Some(_) => lbl.push(3),
    }

    for i in &index[1..] {
        println!("尝试打开{}",format!("{}{}", XML_PATH, format!("{:0>6}", i) + ".xml"));
        annotation = Annotation::from_file(format!("{}{}", XML_PATH, format!("{:0>6}", i) + ".xml")).expect("读取标注文件错误");
        data.extend(annotation.get_path()).expect("读取图像错误");

        match annotation.get_name().get(1) {
            None => lbl.push(annotation.get_name()[0]),
            Some(_) => lbl.push(3),
        }
    }
    println!("Data length is {}", data.get_data().len());
    (data, lbl)
}

fn train_tensor(i: Vec<i64>) -> (Tensor, Tensor) {
    // let size = i.len();
    let (train_data, train_lbl) = data_and_lbl(i);
    let train_data = image_to_tensor(train_data.get_data(), train_data.get_data().len()/train_data.len(), RESIZE, RESIZE);
    let train_lbl_len = train_lbl.len();
    let train_lbl = labels_to_tensor(train_lbl, train_lbl_len, 1);

    (train_data, train_lbl)
}

fn main() {
    // let (train_data, _train_lbl) = data_and_lbl(0, BATCH_SIZE as usize);
    // let _train_data = image_to_tensor(train_data.get_data(), train_data.get_data().len()/(RESIZE*RESIZE), RESIZE, RESIZE);

    // 创建变量保存CUDA是否可用
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();
    let n_it = (TRAIN_SIZE as i64) / BATCH_SIZE;
    println!("尝试创建卷积神经网络");
    let net = ConvNN::new(&vs.root());
    for epoch in 1..N_EPOCHS {
        println!("开始第{}次迭代", epoch);
        // let (train_data, train_lbl) = train_tensor(i);
            // generate random idxs for batch size 
            // run all the images divided in batches  -> for loop
        for _i in 1..n_it {
            println!("尝试生成随机梯度下降索引");
            let batch_idxs = generate_random_index(TRAIN_SIZE as i64, BATCH_SIZE);
            println!("尝试选择一批数据并将其转换为浮点类型");
            let (train_data, train_lbl) = train_tensor(batch_idxs.iter::<i64>().unwrap().collect());
            // let batch_images = train_data.index_select(0, &batch_idxs).to_device(vs.device()).to_kind(Kind::Float);
            let batch_images = train_data.to_device(vs.device()).to_kind(Kind::Float);
            println!("尝试选择一批标注并将其转换为整数类型");
            let batch_lbls = train_lbl.to_device(vs.device()).to_kind(Kind::Int64);
            println!("尝试对批量图像数据进行前向传播，并计算交叉熵损失");
            // compute the loss 
            let loss = net.forward_t(&batch_images, true).cross_entropy_for_logits(&batch_lbls);
            println!("尝试使用优化器对神经网络参数进行更新");
            opt.backward_step(&loss);
        }
        {
            let (val_data, val_lbl) = data_and_lbl((TRAIN_SIZE as i64 +1 .. ALL_SIZE as i64 +1).collect());
            let val_data = image_to_tensor(val_data.get_data(), val_data.get_data().len()/val_data.len(), RESIZE, RESIZE);
            let val_lbl_len = val_lbl.len();
            let val_lbl = labels_to_tensor(val_lbl, val_lbl_len, 1);
            // compute accuracy 
            let val_accuracy =
                net.batch_accuracy_for_logits(&val_data, &val_lbl, vs.device(), 1024);
            println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * val_accuracy,);
        }
    }
}