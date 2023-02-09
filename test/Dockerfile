FROM continuumio/anaconda3
EXPOSE 5000
WORKDIR /app/
COPY ./img_api.py /app/img_api.py
COPY ./requirement.txt /app/requirement.txt
COPY ./model.h5 /app/model.h5
RUN pip install -r requirement.txt
CMD ["python", "img_api.py"]