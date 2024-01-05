FROM spikeinterface/kilosort2_5-compiled-base:0.2.0

# Install Dendro from source, for now
WORKDIR /src
RUN git clone https://github.com/flatironinstitute/dendro.git && \
    cd dendro/python && \
    pip install -e .

# Install spikeinterface-pipelines from source, for now
RUN git clone https://github.com/SpikeInterface/spikeinterface_pipelines.git && \
    cd spikeinterface_pipelines && \
    git checkout dev && \
    pip install -e .

# Install spikeinterface from source, for now
RUN git clone https://github.com/SpikeInterface/spikeinterface.git && \
    cd spikeinterface && \
    # git checkout dev && \
    pip install -e .[full]

# Copy files into the container
WORKDIR /app
COPY *.py ./

# Install requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Set the entrypoint
ENTRYPOINT ["python", "/app/main.py"]