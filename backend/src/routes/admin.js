// Admin routes: /api/admin/
const express = require('express');
const { verifyToken, requireRole } = require('../middleware/authMiddleware');
const {
  getDashboard,
  getClients,
  updateClientStatus,
  getRounds,
  getRoundDetails,
  createRound,
  updateRound,
  startRound,
  cancelRound,
  deleteRound,
  checkRoundDependencies,
  archiveRound,
  getSubmissions,
  reviewSubmission,
  getAllSubmissions,
  getAnalytics,
  getAuditLogs,
  getSystemHealth,
  checkDatasets,
  getRoundDatasetAudit,
} = require('../controllers/adminController');
const { getAdminTrainingMonitor, streamAdminTrainingProgress } = require('../controllers/trainingMonitorController');

const router = express.Router();

// All admin routes require authentication and ADMIN role
router.use(verifyToken, requireRole('ADMIN'));

router.get('/dashboard', getDashboard);
router.get('/system-health', getSystemHealth);
router.get('/training-monitor', getAdminTrainingMonitor);
router.get('/training-progress', streamAdminTrainingProgress);
router.get('/clients', getClients);
router.patch('/clients/:clientId/status', updateClientStatus);

// Dataset Pre-Training Audit Endpoints
router.post('/dataset-audit', checkDatasets);
router.get('/rounds/:roundId/dataset-audit', getRoundDatasetAudit);

// Round Management Endpoints
router.get('/rounds', getRounds);
router.post('/rounds', createRound);
router.get('/rounds/:roundId', getRoundDetails);
router.put('/rounds/:roundId', updateRound);
router.post('/rounds/:roundId/start', startRound);
router.post('/rounds/:roundId/cancel', cancelRound);
router.get('/rounds/:roundId/dependencies', checkRoundDependencies);
router.post('/rounds/:roundId/archive', archiveRound);
router.delete('/rounds/:roundId', deleteRound);

router.get('/submissions', getAllSubmissions);
router.get('/rounds/:roundId/submissions', getSubmissions);
router.patch('/submissions/:submissionId/review', reviewSubmission);
router.get('/analytics', getAnalytics);
router.get('/audit-logs', getAuditLogs);

module.exports = router;
