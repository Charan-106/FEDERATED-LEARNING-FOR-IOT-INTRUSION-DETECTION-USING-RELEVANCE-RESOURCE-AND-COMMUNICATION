// ARC routes: /api/arc/
const express = require('express');
const { verifyToken, requireRole } = require('../middleware/authMiddleware');
const {
  getScores,
  getConfig,
  getConfigHistory,
  updateConfig,
  recalculateScores,
} = require('../controllers/arcController');

const router = express.Router();

router.get('/scores', verifyToken, getScores);
router.get('/config', verifyToken, getConfig);
router.get('/configs', verifyToken, requireRole('ADMIN'), getConfigHistory);
router.put('/config', verifyToken, requireRole('ADMIN'), updateConfig);
router.post('/recalculate', verifyToken, requireRole('ADMIN'), recalculateScores);

module.exports = router;
