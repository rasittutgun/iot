FROM node:16 AS build

# Create a directory where our app will be placed
RUN mkdir -p /app

# Change directory so that our commands run inside this new directory
WORKDIR /app

# Copy dependency definitions
COPY package*.json /app/

# Install dependecies
RUN npm install

# Get all the code needed to run the app
COPY . /app/

# Serve the app
RUN npm run build --prod

# Expose the port the app runs in
EXPOSE 4200

# Serve the app
CMD ["npm", "start"]



#FROM nginx:alpine
#COPY nginx.conf /etc/nginx/nginx.conf

#RUN rm -rf /usr/share/nginx/html/*
#COPY --from=build /app/dist/iotApp /usr/share/nginx/html

# Expose the port the app runs in
#EXPOSE 4200 80

#ENTRYPOINT ["nginx", "-g", "daemon off;"]