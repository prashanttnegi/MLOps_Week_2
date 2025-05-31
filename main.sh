! pip3 install --upgrade --quiet  google-cloud-aiplatform

! gsutil mb -l "us-central1" -p "sharp-cursor-461116-t1" "gs://mlops-course-sharp-cursor-461116-t1-week2"

python main.py

!gsutil cp artifacts/model.joblib "gs://mlops-course-sharp-cursor-461116-t1-week2"/"my-models/iris-classifier-week-2"/