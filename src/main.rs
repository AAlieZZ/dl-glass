mod tensor;
mod annotation;
mod image;
// mod yolo;
mod darknet;

use tensor::{image_to_tensor, labels_to_tensor, generate_random_index};
use crate::image::{Image, RESIZE};
use annotation::Annotation;
use tch::{nn, nn::{ModuleT, OptimizerConfig}, Device, Kind, Tensor};
// use yolo::CONFIG_NAME;
use rand::Rng;

const N_EPOCHS: i64 = 10; // 迭代次数
const TRAIN_SIZE: usize = 6760;
const ALL_SIZE: usize = 8436;
const BATCH_SIZE: i64 = 64;
const XML_PATH: &str = "/home/aliez/dl-glass/GlassCoverDefectDataset/GlassCover_datset/Annotations/";
const CONFIG_NAME: &str = "/home/aliez/dl-glass/src/yolo-v3.cfg";

// 根据索引读取标注文件，返回图像和标注
fn data_and_lbl(index: Vec<i64>) -> (Image, Vec<f32>) {
    // println!("尝试打开{}",format!("{}{}", XML_PATH, format!("{:0>6}", index[0]) + ".xml"));
    let mut annotation = Annotation::from_file(format!("{}{}", XML_PATH, format!("{:0>6}", index[0]) + ".xml")).expect("读取标注文件错误");
    let mut x = rand::thread_rng().gen_range(0, annotation.get_xmin().len());
    let mut data = Image::new(annotation.get_path()).expect("读取图像错误");

    let mut lbl = Vec::new();
    lbl.push(annotation.get_name()[x] as f32);  //类别
    lbl.push((annotation.get_xmax()[x]-annotation.get_xmin()[x])/2.0);  //中心点 x 坐标
    lbl.push((annotation.get_ymax()[x]-annotation.get_ymin()[x])/2.0);  //中心点 y 坐标
    lbl.push(annotation.get_xmax()[x] - annotation.get_xmin()[x]);  //宽度
    lbl.push(annotation.get_ymax()[x] - annotation.get_ymin()[x]);  //高度

    for i in &index[1..] {
        // println!("尝试打开{}",format!("{}{}", XML_PATH, format!("{:0>6}", i) + ".xml"));
        annotation = Annotation::from_file(format!("{}{}", XML_PATH, format!("{:0>6}", i) + ".xml")).expect("读取标注文件错误");
        x = rand::thread_rng().gen_range(0, annotation.get_xmin().len());
        data.extend(annotation.get_path()).expect("读取图像错误");

        lbl.push(annotation.get_name()[x] as f32);  //类别
        lbl.push((annotation.get_xmax()[x]-annotation.get_xmin()[x])/2.0);  //中心点 x 坐标
        lbl.push((annotation.get_ymax()[x]-annotation.get_ymin()[x])/2.0);  //中心点 y 坐标
        lbl.push(annotation.get_xmax()[x] - annotation.get_xmin()[x]);  //宽度
        lbl.push(annotation.get_ymax()[x] - annotation.get_ymin()[x]);  //高度
    }
    println!("Data length is {}", data.get_data().len());
    (data, lbl)
}

fn train_tensor(i: Vec<i64>) -> (Tensor, Tensor) {
    // let size = i.len();
    let (train_data, train_lbl) = data_and_lbl(i);
    let train_data = image_to_tensor(train_data.get_data(), train_data.get_data().len()/train_data.len(), RESIZE, RESIZE);
    let train_lbl = labels_to_tensor(train_lbl, BATCH_SIZE as usize, 5);

    (train_data, train_lbl)
}

fn test(net: &nn::FuncT, vs: &nn::VarStore, epoch: i64) {
    let (val_data, val_lbl) = data_and_lbl((TRAIN_SIZE as i64 +1 .. ALL_SIZE as i64 +1).collect());
    let val_data = image_to_tensor(val_data.get_data(), val_data.get_data().len()/val_data.len(), RESIZE, RESIZE);
    let val_lbl = labels_to_tensor(val_lbl, BATCH_SIZE as usize, 5);
    // compute accuracy 
    let val_accuracy =
        net.batch_accuracy_for_logits(&val_data, &val_lbl, vs.device(), 128);
    println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * val_accuracy,);
}

fn main() {
    // let (train_data, _train_lbl) = data_and_lbl(0, BATCH_SIZE as usize);
    // let _train_data = image_to_tensor(train_data.get_data(), train_data.get_data().len()/(RESIZE*RESIZE), RESIZE, RESIZE);

    // 创建变量保存CUDA是否可用
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();
    let darknet = darknet::parse_config(CONFIG_NAME).unwrap();
    let n_it = (TRAIN_SIZE as i64) / BATCH_SIZE;
    println!("尝试创建卷积神经网络");
    let model = darknet.build_model(&vs.root()).unwrap();
    for epoch in 1..N_EPOCHS {
        println!("开始第{}次迭代", epoch);
        // let (train_data, train_lbl) = train_tensor(i);
            // generate random idxs for batch size 
            // run all the images divided in batches  -> for loop
        for i in 1..n_it {
            println!("（{}，{}）尝试生成随机梯度下降索引", epoch, i);
            let batch_idxs = generate_random_index(TRAIN_SIZE as i64, BATCH_SIZE);
            println!("（{}，{}）尝试选择一批数据并将其转换为浮点类型", epoch, i);
            let (train_data, train_lbl) = train_tensor(batch_idxs.iter::<i64>().unwrap().collect());
            // let batch_images = train_data.index_select(0, &batch_idxs).to_device(vs.device()).to_kind(Kind::Float);
            let batch_images = train_data.to_device(vs.device()).to_kind(Kind::Float);
            println!("（{}，{}）尝试选择一批标注", epoch, i);
            let batch_lbls = train_lbl.to_device(vs.device()).to_kind(Kind::Float);
            println!("（{}，{}）尝试对批量图像数据进行前向传播，并计算交叉熵损失", epoch, i);
            // compute the loss 
            let loss = model.forward_t(&batch_images, true).cross_entropy_for_logits(&batch_lbls);
            println!("（{}，{}）尝试使用优化器对神经网络参数进行更新", epoch, i);
            opt.backward_step(&loss);
        }
        test(&model, &vs, epoch);
    }
}