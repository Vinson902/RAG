# docker/postgres/Dockerfile
FROM pgvector/pgvector:pg16

# Copy custom configuration files
COPY postgresql.conf /etc/postgresql/postgresql.conf
COPY pg_hba.conf /etc/postgresql/pg_hba.conf
COPY init-db.sql /docker-entrypoint-initdb.d/

# Set environment variables to match your pi-cluster setup
ENV POSTGRES_DB=ragdb
ENV POSTGRES_USER=raguser
ENV POSTGRES_PASSWORD=ragpass

# Override default config file location
ENV POSTGRES_CONFIG_FILE=/etc/postgresql/postgresql.conf
ENV POSTGRES_HBA_FILE=/etc/postgresql/pg_hba.conf

# Use custom configuration
CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf", "-c", "hba_file=/etc/postgresql/pg_hba.conf"]

EXPOSE 5432