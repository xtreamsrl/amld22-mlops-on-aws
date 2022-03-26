FROM python:3.9-slim

# Install all dependencies without cache. In a real-world project you should always pin the version of the libraries.
RUN pip install --no-cache numpy pandas sktime flask gunicorn pyarrow fastparquet

# Keeps Python from buffering our standard output stream
ENV PYTHONUNBUFFERED=TRUE
# Keeps Python from writing the .pyc files which are unnecessary in this case
ENV PYTHONDONTWRITEBYTECODE=TRUE
# Update PATH so that the train and serve programs are found when the container is invoked
ENV PATH="/opt/program:${PATH}"

COPY persistence/train.py /usr/bin/train
COPY persistence/serve.py /usr/bin/serve
RUN chmod 755 /usr/bin/train /usr/bin/serve
