# **Face_recorgnize Documentation**

# **Setup GPU**

• conda install -c conda-forge cudatoolkit=10.2 cudnn=8.1.0

# **Setup tensorflow GPU**

## ***Environment Setup***


• Open a terminal from the local directory and run ```conda create -n face_recorgnize python=3.7.16``` to create the environment

• Wait for the command to finish and activate the environment by running ```conda activate face_recorgnize```

• Run ```conda env update --file environment.yml --prune```

• Run ```inv upgrade-req``` to upgrade the environment.yml based on your current environment list of package.

## ***How to train and run model***

• Get new user data data: ```inv gui```: click **Đăng ký** to register new user, **Xóa dữ liệu học sinh** to remove existing user. 

• Convert raw images based on model requirement and train model by: ```inv train```

• Run test with: ```inv checkface```

Raw train: ```python src/classifier.py TRAIN Dataset/FaceData/processed/train Models/20180402-114759.pb facemodel.pkl --batch_size 256```





