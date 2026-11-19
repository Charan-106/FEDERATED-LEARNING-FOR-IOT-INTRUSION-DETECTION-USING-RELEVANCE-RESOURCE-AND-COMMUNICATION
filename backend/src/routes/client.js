// Client routes: /api/client/
const express = require('express');
const { verifyToken, requireRole } = require('../middleware/authMiddleware');
const upload = require('../middleware/uploadMiddleware');
const {
  getDashboard,
  getGlobalModel,
  downloadGlobalModel,
  submitUpdate,
  getSubmissions,
  getArcHistory,
  getDatasets,
  runLocalTraining,
  submitTrainedUpdate,
  getClientConfig,
  updateClientConfig,
} = require('../controllers/clientController');
const {
  getClientTrainingMonitor,
  getClientTrainingStatus,
  streamClientTrainingProgress,
} = require('../controllers/trainingMonitorController');

const router = express.Router();

// All client routes require authentication and CLIENT role
router.use(verifyToken, requireRole('CLIENT'));

router.get('/dashboard', getDashboard);
router.get('/model', getGlobalModel);
router.get('/model/download', downloadGlobalModel);
router.get('/datasets', getDatasets);
router.get('/config', getClientConfig);
router.put('/config', updateClientConfig);
router.get('/training-monitor', getClientTrainingMonitor);
router.get('/training-status', getClientTrainingStatus);
router.get('/training-progress', streamClientTrainingProgress);
router.post('/train', runLocalTraining);
router.post('/submit-trained', submitTrainedUpdate);
router.post('/submit', upload.single('modelFile'), submitUpdate);
router.get('/submissions', getSubmissions);
router.get('/arc-history', getArcHistory);

module.exports = router;
