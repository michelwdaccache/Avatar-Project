# Use a base image for multi-stage builds
FROM node:20.11.1-alpine as frontend_builder
FROM python:3.10-slim as backend_builder

# Stage for the frontend
FROM frontend_builder as frontend
WORKDIR /app
COPY ./frontend/package*.json ./
RUN npm install
COPY ./frontend ./
EXPOSE 3000
CMD ["npm", "run", "dev"]

# Stage for the backend
FROM backend_builder as backend
WORKDIR /app
COPY ./backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./backend ./
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]