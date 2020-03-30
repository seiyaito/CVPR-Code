# CVPR2020-Code

CVPR 2020の論文及びオープンソースプロジェクトのコレクションです.そして,もし問題があればissueに提出して,CVPR 2020のオープンソースプロジェクトを共有することも歓迎します

- [画像分類](#Image-Classification)
- [物体検出](#Object-Detection)
- [3D物体検出](#3D-Object-Detection)
- [動画物体検出](#Video-Object-Detection)
- [物体追跡](#Object-Tracking)
- [セマンティックセグメンテーション](#Semantic-Segmentation)
- [インスタンスセグメンテーション](#Instance-Segmentation)
- [動画物体分割](#VOS)
- [NAS](#NAS)
- [GAN](#GAN)
- [Re-ID](#Re-ID)
- [3D点群（セマンティックセグメンテーション等）](#3D-PointCloud)
- [顔識別](#Face-Recognition)
- [顔検出](#Face-Detection)
- [Face Anti-spoofing](#FAS)
- [顔表情識別](#Facial-Expression-Recognition)
- [Face Rotation](#Face-Rotation)
- [人体姿勢推定](#Human-Pose-Estimation)
- [シーンテキスト検出](#Scene-Text-Detection)
- [シーンテキスト識別](#Scene-Text-Recognition)
- [モデル圧縮](#Model-Compression)
- [モデル枝刈り](#Model-Pruning)
- [行動識別](#Action-Recognition)
- [群衆カウント](#Crowd-Counting)
- [深度推定](#Depth-Estimation)
- [6D物体姿勢推定](#6DOF)
- [手势推定](#Hand-Pose)
- [画像修復](#Denoising)
- [デブラー](#Deblurring)
- [特征点検出与描述](#Feature)
- [VQA](#VQA)
- [視覚言語ナビゲーション](#VLN)
- [動画圧縮](#Video-Compression)
- [動画補間](#Video-Frame-Interpolation)
- [スタイル変換](#Style-Transfer)
- [Human-Object Interaction (HOI)検出](#HOI)
- [行動軌跡予測](#HTP)
- [モーション予測](#Motion-Predication)
- [データセット](#Datasets)
- [その他](#Others)
- [採択されたか不明](#Not-Sure)

<a name="Image-Classification"></a>

# 画像分類

**Spatially Attentive Output Layer for Image Classification**

- 論文：未公開

- コード： https://github.com/ildoonet/spatially-attentive-output-layer 

<a name="Object-Detection"></a>

# 物体検出

**Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection**

- 論文：https://arxiv.org/abs/1912.02424 
- コード：https://github.com/sfzhang15/ATSS

**BiDet: An Efficient Binarized Object Detector**

- 論文：https://arxiv.org/abs/2003.03961 
- コード：https://github.com/ZiweiWangTHU/BiDet

**Harmonizing Transferability and Discriminability for Adapting Object Detectors**

- 論文：https://arxiv.org/abs/2003.06297
- コード：https://github.com/chaoqichen/HTCN

**CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection**

- 論文：https://arxiv.org/abs/2003.09119
- コード：https://github.com/KiveeDong/CentripetalNet

**Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection**

- 論文：https://arxiv.org/abs/2003.11818
- コード：https://github.com/ggjy/HitDet.pytorch

**EfficientDet: Scalable and Efficient Object Detection**

- 論文：https://arxiv.org/abs/1911.09070
- コード：https://github.com/google/automl/tree/master/efficientdet 

<a name="3D-Object-Detection"></a>

# 3D物体検出

**PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection**

- 論文：https://arxiv.org/abs/1912.13192

- コード：https://github.com/sshaoshuai/PV-RCNN

**Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud**

- 論文：https://arxiv.org/abs/2003.01251 
- コード：https://github.com/WeijingShi/Point-GNN 

<a name="Video-Object-Detection"></a>

# 動画物体検出

**Memory Enhanced Global-Local Aggregation for Video Object Detection**

論文：https://arxiv.org/abs/2003.12063

コード：https://github.com/Scalsol/mega.pytorch

<a name="Object-Tracking"></a>

# 物体追跡

**Probabilistic Regression for Visual Tracking**

- 論文：https://arxiv.org/abs/2003.12565
- コード：https://github.com/visionml/pytracking

**MAST: A Memory-Augmented Self-supervised Tracker**

- 論文：https://arxiv.org/abs/2002.07793
- コード：https://github.com/zlai0/MAST

**Siamese Box Adaptive Network for Visual Tracking**

- 論文：https://arxiv.org/abs/2003.06761

- コード：https://github.com/hqucv/siamban

<a name="Semantic-Segmentation"></a>

# セマンティックセグメンテーション

**Cars Can't Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks**

- 論文：https://arxiv.org/abs/2003.05128
- コード：https://github.com/shachoi/HANet

**Learning Dynamic Routing for Semantic Segmentation**

- 論文：https://arxiv.org/abs/2003.10401

- コード：https://github.com/yanwei-li/DynamicRouting

<a name="Instance-Segmentation"></a>

# インスタンスセグメンテーション

**PolarMask: Single Shot Instance Segmentation with Polar Representation**

- 論文：https://arxiv.org/abs/1909.13226 
- コード：https://github.com/xieenze/PolarMask 
- 解説：https://zhuanlan.zhihu.com/p/84890413 

**CenterMask : Real-Time Anchor-Free Instance Segmentation**

- 論文：https://arxiv.org/abs/1911.06667 
- コード：https://github.com/youngwanLEE/CenterMask 

**Deep Snake for Real-Time Instance Segmentation**

- 論文：https://arxiv.org/abs/2001.01629
- コード：https://github.com/zju3dv/snake

**Mask Encoding for Single Shot Instance Segmentation**

- 論文：https://arxiv.org/abs/2003.11712

- コード：https://github.com/aim-uofa/AdelaiDet

<a name="VOS"></a>

# 動画物体分割

**State-Aware Tracker for Real-Time Video Object Segmentation**

- 論文：https://arxiv.org/abs/2003.00482

- コード：https://github.com/MegviiDetection/video_analyst

**Learning Fast and Robust Target Models for Video Object Segmentation**

- 論文：https://arxiv.org/abs/2003.00908 
- コード：https://github.com/andr345/frtm-vos

**Learning Video Object Segmentation from Unlabeled Videos**

- 論文：https://arxiv.org/abs/2003.05020
- コード：https://github.com/carrierlxk/MuG

<a name="NAS"></a>

# NAS

**Rethinking Performance Estimation in Neural Architecture Search**

- 論文：準備中
- コード：https://github.com/zhengxiawu/rethinking_performance_estimation_in_NAS
- 解説：https://www.zhihu.com/question/372070853/answer/1035234510

**CARS: Continuous Evolution for Efficient Neural Architecture Search**

- 論文：https://arxiv.org/abs/1909.04977 
- コード（公開前）：https://github.com/huawei-noah/CARS 

<a name="GAN"></a>

# GAN

**Learning to Cartoonize Using White-box Cartoon Representations**

- 論文：https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf

- ホームページ：https://systemerrorwang.github.io/White-box-Cartoonization/
- コード：https://github.com/SystemErrorWang/White-box-Cartoonization
- 解説：https://zhuanlan.zhihu.com/p/117422157
- Demo動画：https://www.bilibili.com/video/av56708333

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**

- 論文：https://arxiv.org/abs/2003.08936

- コード：https://github.com/mit-han-lab/gan-compression

**Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions**

- 論文：https://arxiv.org/abs/2003.01826 
- コード：https://github.com/cc-hpc-itwm/UpConv 

<a name="Re-ID"></a>

# Re-ID

**Weakly supervised discriminative feature learning with state information for person identification**

- 論文：https://arxiv.org/abs/2002.11939 
- コード：https://github.com/KovenYu/state-information 

<a name="3D-PointCloud"></a>

# 3D点群（セマンティックセグメンテーション等）

## 3D点群畳み込み

**Grid-GCN for Fast and Scalable Point Cloud Learning**

- 論文：https://arxiv.org/abs/1912.02984

- コード：https://github.com/Xharlie/Grid-GCN

**FPConv: Learning Local Flattening for Point Convolution**

- 論文：https://arxiv.org/abs/2002.10701
- コード：https://github.com/lyqun/FPConv

## 3D点群セマンティックセグメンテーション

**Learning to Segment 3D Point Clouds in 2D Image Space**

- 論文：https://arxiv.org/abs/2003.05593

- コード：https://github.com/WPI-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space

## 3D点群レジストレーション

**D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features**

- 論文：https://arxiv.org/abs/2003.03164
- コード：https://github.com/XuyangBai/D3Feat

**RPM-Net: Robust Point Matching using Learned Features**

- 論文：未公開

- コード：https://github.com/yewzijian/RPMNet 

<a name="Face-Recognition"></a>

# 顔識別

**Learning Meta Face Recognition in Unseen Domains**

- 論文：https://arxiv.org/abs/2003.07733
- コード：https://github.com/cleardusk/MFR
- 解説：https://mp.weixin.qq.com/s/YZoEnjpnlvb90qSI3xdJqQ 

<a name="Face-Detection"></a>

# 顔検出

<a name="FAS"></a>

# Face Anti-spoofing

**Searching Central Difference Convolutional Networks for Face Anti-Spoofing**

- 論文：https://arxiv.org/abs/2003.04092

- コード：https://github.com/ZitongYu/CDCN

<a name="Facial-Expression-Recognition"></a>

# 顔表情識別

**Suppressing Uncertainties for Large-Scale Facial Expression Recognition**

- 論文：https://arxiv.org/abs/2002.10392 

- コード（公開前）：https://github.com/kaiwang960112/Self-Cure-Network 

<a name="Face-Rotation"></a>

# Face Rotation

**Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images**

- 論文：https://arxiv.org/abs/2003.08124

- コード：https://github.com/Hangz-nju-cuhk/Rotate-and-Render

<a name="Human-Pose-Estimation"></a>

# 人体姿勢推定

## 2D人体姿勢推定

**The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation**

- 論文：https://arxiv.org/abs/1911.07524 
- コード：https://github.com/HuangJunJie2017/UDP-Pose
- 解説：https://zhuanlan.zhihu.com/p/92525039

**Distribution-Aware Coordinate Representation for Human Pose Estimation**

- ホームページ：https://ilovepose.github.io/coco/ 

- 論文：https://arxiv.org/abs/1910.06278 

- コード：https://github.com/ilovepose/DarkPose 

## 3D人体姿勢推定

**Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation**

- 論文：なし

- コード：https://github.com/anonymous-goat/CVPR-2020 

**VIBE: Video Inference for Human Body Pose and Shape Estimation**

- 論文：https://arxiv.org/abs/1912.05656 
- コード：https://github.com/mkocabas/VIBE

**Back to the Future: Joint Aware Temporal Deep Learning 3D Human Pose Estimation**

- 論文：https://arxiv.org/abs/2002.11251 
- コード：https://github.com/vnmr/JointVideoPose3D

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**

- 論文：https://arxiv.org/abs/2003.03972
- データセット：なし

<a name="Scene-Text-Detection"></a>

# 点群

## 点群分類

**PointAugment: an Auto-Augmentation Framework for Point Cloud Classification**

- 論文：https://arxiv.org/abs/2002.10876 
- コード（公開前）： https://github.com/liruihui/PointAugment/ 

# シーンテキスト検出

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- 論文：https://arxiv.org/abs/2003.10608
- コード和データセット：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**

- 論文：https://arxiv.org/abs/2002.10200 
- コード（公開前）：https://github.com/Yuliang-Liu/bezier_curve_text_spotting
- コード（公開前）：https://github.com/aim-uofa/adet

**Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection**

- 論文：https://arxiv.org/abs/2003.07493

- コード：https://github.com/GXYM/DRRG

<a name="Scene-Text-Recognition"></a>

# シーンテキスト識別

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- 論文：https://arxiv.org/abs/2003.10608
- コード和データセット：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**

- 論文：https://arxiv.org/abs/2002.10200 
- コード（公開前）：https://github.com/aim-uofa/adet

**Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition**

- 論文：https://arxiv.org/abs/2003.06606

- コード：https://github.com/Canjie-Luo/Text-Image-Augmentation

<a name="Super-Resolution"></a>

# 超解像

## 動画超解像

**Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution**

- 論文：https://arxiv.org/abs/2002.11616 
- コード：https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020 

<a name="Model-Compression"></a>

# モデル圧縮

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**

- 論文：https://arxiv.org/abs/2003.08936

- コード：https://github.com/mit-han-lab/gan-compression

**Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression**

- 論文：https://arxiv.org/abs/2003.08935

- コード：https://github.com/ofsoundof/group_sparsity

<a name="Model-Pruning"></a>

# モデル枝刈り

**HRank: Filter Pruning using High-Rank Feature Map**

- 論文：http://arxiv.org/abs/2002.10179
- コード：https://github.com/lmbxmu/HRank 

<a name="Action-Recognition"></a>

# 行動識別

**Temporal Pyramid Network for Action Recognition**

- 論文：準備中
- コード：https://github.com/limbo0000/TPN

<a name="Crowd-Counting"></a>

# 群衆カウント

<a name="Depth-Estimation"></a>

# 深度推定

## 単眼深度推定

**Domain Decluttering: Simplifying Images to Mitigate Synthetic-Real Domain Shift and Improve Depth Estimation**

- 論文：https://arxiv.org/abs/2002.12114
- コード：https://github.com/yzhao520/ARC

<a name="6DOF"></a>

# 6D物体姿勢推定

**G2L-Net: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features**

- 論文：https://arxiv.org/abs/2003.11089

- コード：https://github.com/DC1991/G2L_Net

<a name="Hand-Pose"></a>

# 手势推定

**Monocular Real-time Hand Shape and Motion Capture using Multi-modal Data**

- 論文：https://arxiv.org/abs/2003.09572

- コード：https://github.com/CalciferZh/minimal-hand

<a name="Denoising"></a>

# 画像修復

**CycleISP: Real Image Restoration via Improved Data Synthesis**

- 論文：https://arxiv.org/abs/2003.07761

- コード：https://github.com/swz30/CycleISP

<a name="Deraining"></a>

# 去雨

**Multi-Scale Progressive Fusion Network for Single Image Deraining**

- 論文：https://arxiv.org/abs/2003.10985

- コード：https://github.com/kuihua/MSPFN

<a name="Deblurring"></a>

# デブラー

## 動画デブラー

**Cascaded Deep Video Deblurring Using Temporal Sharpness Prior**

- ホームページ：https://csbhr.github.io/projects/cdvd-tsp/index.html 
- 論文：準備中
- コード：https://github.com/csbhr/CDVD-TSP

<a name="Feature"></a>

# 特征点検出与描述

**ASLFeat: Learning Local Features of Accurate Shape and Localization**

- 論文：https://arxiv.org/abs/2003.10071

- コード：https://github.com/lzx551402/aslfeat

<a name="VQA"></a>

# VQA

**VC R-CNN：Visual Commonsense R-CNN** 

- 論文：https://arxiv.org/abs/2002.12204
- コード：https://github.com/Wangt-CN/VC-R-CNN

<a name="VLN"></a>

# 視覚言語ナビゲーション

**Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training**

- 論文：https://arxiv.org/abs/2002.10638
- コード（公開前）：https://github.com/weituo12321/PREVALENT

<a name="Video-Compression"></a>

# 動画圧縮

**Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement**

- 論文：https://arxiv.org/abs/2003.01966 
- コード：https://github.com/RenYang-home/HLVC

<a name="Video-Frame-Interpolation"></a>

# 動画補間

**Softmax Splatting for Video Frame Interpolation**

- ホームページ：http://sniklaus.com/papers/softsplat
- 論文：https://arxiv.org/abs/2003.05534
- コード：https://github.com/sniklaus/softmax-splatting

<a name="Style-Transfer"></a>

# スタイル変換

**Collaborative Distillation for Ultra-Resolution Universal Style Transfer**

- 論文：https://arxiv.org/abs/2003.08436

- コード：https://github.com/mingsun-tse/collaborative-distillation

<a name="HOI"></a>

# Human-Object Interaction (HOT)検出

**Cascaded Human-Object Interaction Recognition**

- 論文：https://arxiv.org/abs/2003.04262

- コード：https://github.com/tfzhou/C-HOI

**VSGNet: Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions**

- 論文：https://arxiv.org/abs/2003.05541
- コード：https://github.com/ASMIftekhar/VSGNet

<a name="HTP"></a>

# 歩行者軌跡予測

**Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction**

- 論文：https://arxiv.org/abs/2002.11927 
- コード：https://github.com/abduallahmohamed/Social-STGCNN 

<a name="Motion-Predication"></a>

# モーション予測

**Collaborative Motion Prediction via Neural Motion Message Passing**

- 論文：https://arxiv.org/abs/2003.06594
- コード：https://github.com/PhyllisH/NMMP

**MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps**

- 論文：https://arxiv.org/abs/2003.06754

- コード：https://github.com/pxiangwu/MotionNet

<a name="Datasets"></a>

# データセット

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- 論文：https://arxiv.org/abs/2003.10608
- コード和データセット：https://github.com/Jyouhou/UnrealText/

**PANDA: A Gigapixel-level Human-centric Video Dataset**

- 論文：https://arxiv.org/abs/2003.04852

- データセット：http://www.panda-dataset.com/

**IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning**

- 論文：https://arxiv.org/abs/2003.02920
- データセット：https://github.com/intra3d2019/IntrA

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**

- 論文：https://arxiv.org/abs/2003.03972
- データセット：なし

<a name="Others"></a>

# その他

**What Deep CNNs Benefit from Global Covariance Pooling: An Optimization Perspective**

- 論文：https://arxiv.org/abs/2003.11241

- コード：https://github.com/ZhangLi-CS/GCP_Optimization

**Look-into-Object: Self-supervised Structure Modeling for Object Recognition**

- 論文：なし
- コード：https://github.com/JDAI-CV/LIO 

 **Video Object Grounding using Semantic Roles in Language Description**

- 論文：https://arxiv.org/abs/2003.10606
- コード：https://github.com/TheShadow29/vognet-pytorch 

**Dynamic Hierarchical Mimicking Towards Consistent Optimization Objectives**

- 論文：https://arxiv.org/abs/2003.10739
- コード：https://github.com/d-li14/DHM 

**SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization**

- 論文：http://www.cs.umd.edu/~yuejiang/papers/SDFDiff.pdf
- コード：https://github.com/YueJiang-nj/CVPR2020-SDFDiff 

**On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location**

- 論文：https://arxiv.org/abs/2003.07064

- コード：https://github.com/oskyhn/CNNs-Without-Borders

**GhostNet: More Features from Cheap Operations**

- 論文：https://arxiv.org/abs/1911.11907

- コード：https://github.com/iamhankai/ghostnet

**AdderNet: Do We Really Need Multiplications in Deep Learning?** 

- 論文：https://arxiv.org/abs/1912.13200 
- コード：https://github.com/huawei-noah/AdderNet

**Deep Image Harmonization via Domain Verification** 

- 論文：https://arxiv.org/abs/1911.13239 
- コード：https://github.com/bcmi/Image_Harmonization_Datasets

**Blurry Video Frame Interpolation**

- 論文：https://arxiv.org/abs/2002.12259 
- コード：https://github.com/laomao0/BIN

**Extremely Dense Point Correspondences using a Learned Feature Descriptor**

- 論文：https://arxiv.org/abs/2003.00619 
- コード：https://github.com/lppllppl920/DenseDescriptorLearning-Pytorch

**Filter Grafting for Deep Neural Networks**

- 論文：https://arxiv.org/abs/2001.05868
- コード：https://github.com/fxmeng/filter-grafting
- 論文解説：https://www.zhihu.com/question/372070853/answer/1041569335

**Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation**

- 論文：https://arxiv.org/abs/2003.02824 
- コード：https://github.com/cmhungsteve/SSTDA

**Detecting Attended Visual Targets in Video**

- 論文：https://arxiv.org/abs/2003.02501 

- コード：https://github.com/ejcgt/attention-target-detection

**Deep Image Spatial Transformation for Person Image Generation**

- 論文：https://arxiv.org/abs/2003.00696 
- コード：https://github.com/RenYurui/Global-Flow-Local-Attention

 **Rethinking Zero-shot Video Classification: End-to-end Training for Realistic Applications** 

- 論文：https://arxiv.org/abs/2003.01455
- コード：https://github.com/bbrattoli/ZeroShotVideoClassification

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/Anonymous20192020/Anonymous_CVPR5767

https://github.com/avirambh/ScopeFlow

https://github.com/csbhr/CDVD-TSP

https://github.com/ymcidence/TBH

https://github.com/yaoyao-liu/mnemonics

https://github.com/meder411/Tangent-Images

https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

https://github.com/sjmoran/deep_local_parametric_filters

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/bermanmaxim/AOWS

https://github.com/dc3ea9f/look-into-object 

<a name="Not-Sure"></a>

# 採択されたか不明

**FADNet: A Fast and Accurate Network for Disparity Estimation**

- 論文：未公開
- コード：https://github.com/HKBU-HPML/FADNet

https://github.com/rFID-submit/RandomFID

https://github.com/JackSyu/AE-MSR

https://github.com/fastconvnets/cvpr2020

https://github.com/aimagelab/meshed-memory-transformer

https://github.com/TWSFar/CRGNet

https://github.com/CVPR-2020/CDARTS

https://github.com/anucvml/ddn-cvprw2020

https://github.com/dl-model-recommend/model-trust

https://github.com/apratimbhattacharyya18/CVPR-2020-Corr-Prior

https://github.com/onetcvpr/O-Net

https://github.com/502463708/Microcalcification_Detection

https://github.com/anonymous-for-review/cvpr-2020-deep-smoke-machine

https://github.com/anonymous-for-review/cvpr-2020-smoke-recognition-dataset

https://github.com/cvpr-nonrigid/dataset

https://github.com/theFool32/PPBA

https://github.com/Realtime-Action-Recognition/Realtime-Action-Recognition

## 注意事項

このリポジトリは [amusi/CVPR2020-Code](https://github.com/amusi/CVPR2020-Code)を日本語訳したものです．

元のリポジトリの内容及び日本語訳は正確でないことがありますのでご注意ください．


## Acknowledgments

Proofread by Jin Jiongxing
