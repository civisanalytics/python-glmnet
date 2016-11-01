FROM ubuntu:14.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && \
    apt-get install -y \
        python3-all \
        python3-dev \
        python3-pip \
        liblapack-dev \
        libatlas-dev \
        gfortran

RUN ln -nsf /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install -U -qq --no-deps "numpy==1.9.3"
RUN pip install -U -qq --no-deps "scipy==0.14.1"
RUN pip install -U -qq --no-deps "cython==0.21.1"
RUN pip install -U -qq --no-deps "scikit-learn==0.18.0"
RUN pip install -U -qq --no-deps "python-dateutil==2.2"
RUN pip install -U -qq --no-deps "pytz"
RUN pip install -U -qq --no-deps "pandas==0.17.1"
RUN pip install -U -qq --no-deps "nose==1.3.7"

COPY . /src/python-glmnet
RUN cd /src/python-glmnet && python setup.py install
