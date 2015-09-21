FROM python:3.4
MAINTAINER Chunhong "Chuck" Yoon <yoon82@slac.stanford.edu>

ENV WORKDIR /scikit-xray
RUN mkdir -p $WORKDIR
COPY . $WORKDIR

RUN pip install -r $WORKDIR/requirements.txt

RUN cd $WORKDIR && \
    python setup.py install

