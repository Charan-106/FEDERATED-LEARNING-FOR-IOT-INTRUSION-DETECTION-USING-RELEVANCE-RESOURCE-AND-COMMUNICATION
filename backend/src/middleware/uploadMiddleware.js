// Multer upload middleware for client model updates
// Accepts only .pt (PyTorch checkpoint) files up to 100MB

const multer = require('multer');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const fs = require('fs');
const config = require('../config');

// Ensure upload directory exists
const uploadDir = config.storage.clientUpdates;
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    // Generate unique filename: uuid + original extension (.pt)
    const ext = path.extname(file.originalname).toLowerCase();
    const uniqueName = `${uuidv4()}${ext}`;
    cb(null, uniqueName);
  },
});

// File filter: accept only .pt files
function fileFilter(req, file, cb) {
  const ext = path.extname(file.originalname).toLowerCase();
  if (ext !== '.pt') {
    const err = new Error('Only .pt (PyTorch checkpoint) files are accepted');
    err.statusCode = 400;
    return cb(err, false);
  }
  cb(null, true);
}

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: config.upload.maxSizeMb * 1024 * 1024, // 100MB in bytes
  },
});

module.exports = upload;
