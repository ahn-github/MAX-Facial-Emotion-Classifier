FROM codait/max-base:v1.1.1

# Fill in these with a link to the bucket containing the model and the model file name
ARG model_bucket=http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/max-facial-emotion-classifier/1.0
ARG model_file=assets.tar.gz

RUN apt-get update && apt-get install libgtk2.0 -y && rm -rf /var/lib/apt/lists/*
RUN wget -nv --show-progress --progress=bar:force:noscroll ${model_bucket}/${model_file} --output-document=/workspace/assets/${model_file} && \
  tar -x -C assets/ -f assets/${model_file} -v && rm assets/${model_file}

COPY requirements.txt /workspace
RUN pip install -r requirements.txt 

COPY . /workspace
RUN md5sum -c md5sums.txt # check file integrity

EXPOSE 5000

CMD python app.py
