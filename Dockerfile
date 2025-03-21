# Use an appropriate base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install supervisord
RUN apt-get update && apt-get install -y supervisor && apt-get clean

# Copy supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the port that the web server will use
EXPOSE 8000

# Set the entry point to supervisord
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
