# facenet_tf2
This program is using facenet solution : https://github.com/davidsandberg/facenet 
This program is adapting the original solution for working with tensorflow 2.x 
This program integrate also the possiblity to aligh images using mtcnn program

-- Extraction et localisation des visages
python src/align/align_dataset_mtcnn.py PERSONS PERSONS_ALIGNED
-- Reconnaissance des visages
$ python facenet\src\classifier.py TRAIN PERSONS_ALIGNED .\model_checkpoints\20180408-102900.pb .\model_checkpoints\my_classifier.pkl


-- Training
cd facenet
$ python .\src\classifier.py TRAIN ..\persons_aligned_gr1 ..\model_checkpoints\20180408-102900.pb ..\model_checkpoints\my_classifier.pkl --batch_size 100

-- real time recognition
cd facenet
$ python contributed\real_time_face_recognition.py

-- offline recognition
cd facenet
$ python contributed\offline_face_recognition.py