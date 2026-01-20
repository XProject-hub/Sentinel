FROM node:20-alpine

WORKDIR /app

# Git commit passed at build time
ARG GIT_COMMIT=unknown
ENV NEXT_PUBLIC_GIT_COMMIT=$GIT_COMMIT

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install

# Copy application
COPY . .

# Build
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Production settings
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

EXPOSE 3000

CMD ["npm", "start"]

