FROM python:3.9

# Install spikeinterface-pipelines from source, for now
RUN git clone https://github.com/SpikeInterface/spikeinterface_pipelines.git && \
    cd spikeinterface_pipelines && \
    # git checkout dev && \
    pip install -e .

# Install spikeinterface
RUN pip install spikeinterface[full]==0.100.0

# Install dendro
RUN pip install dendro==0.2.2

# Copy files into the container
WORKDIR /app
COPY /common /app/common
COPY si_preprocessing_dev/*.py /app/
COPY si_preprocessing_dev/nwbdendroextractors/*.py /app/nwbdendroextractors/
COPY si_preprocessing_dev/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Give execution permission to /app/main.py
RUN chmod +x /app/main.py
