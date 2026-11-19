// Aggregation routes: /api/aggregation/
const express = require('express');
const { verifyToken, requireRole } = require('../middleware/authMiddleware');
const {
  previewWeights,
  mergeModels,
  getHistory,
  getRoundAggregation,
} = require('../controllers/aggregationController');

const router = express.Router();

router.use(verifyToken, requireRole('ADMIN'));

router.get('/preview', previewWeights);
router.post('/preview', previewWeights);
router.post('/merge', mergeModels);
router.get('/history', getHistory);
router.get('/round/:roundId', getRoundAggregation);

module.exports = router;
