FROM continuumio/anaconda3
EXPOSE 8000
RUN apt-get update && apt-get install -y apache2 \
    apache2-dev \   
    vim \
 && apt-get clean \
 && apt-get autoremove \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /var/www/Img_Rec_API/
COPY ./Image_Recognition.wsgi /var/www/Img_Rec_API/Image_Recognition.wsgi
COPY ./Img_Rec_API.py /var/www/Img_Rec_API/Img_Rec_API.py
COPY ./requirement.txt /var/www/Img_Rec_API/requirement.txt
RUN pip install -r requirement.txt
RUN /opt/conda/bin/mod_wsgi-express install-module
RUN mod_wsgi-express setup-server Image_Recognition.wsgi --port=8000 \
    --user www-data --group www-data \
    --server-root=/etc/mod_wsgi-express-80
CMD /etc/mod_wsgi-express-80/apachectl start -D FOREGROUND