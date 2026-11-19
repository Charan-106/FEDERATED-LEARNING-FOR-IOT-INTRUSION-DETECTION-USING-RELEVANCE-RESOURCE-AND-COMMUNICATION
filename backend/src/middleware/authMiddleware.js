// JWT Authentication Middleware for ARC-FL Backend
// Verifies Bearer tokens and attaches decoded user to req.user

const jwt = require('jsonwebtoken');
const config = require('../config');
const prisma = require('../config/database');

/**
 * verifyToken - Express middleware that validates JWT Bearer tokens.
 * Attaches decoded user object to req.user on success.
 */
async function verifyToken(req, res, next) {
  try {
    const authHeader = req.headers['authorization'];
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'No Bearer token provided',
        timestamp: new Date().toISOString(),
      });
    }

    const token = authHeader.split(' ')[1];
    let decoded;
    try {
      decoded = jwt.verify(token, config.jwt.secret);
    } catch (err) {
      const message = err.name === 'TokenExpiredError'
        ? 'Token has expired'
        : 'Invalid token';
      return res.status(401).json({
        error: 'Unauthorized',
        message,
        timestamp: new Date().toISOString(),
      });
    }

    // Fetch fresh user record to catch role changes or account deletion
    const user = await prisma.user.findUnique({
      where: { id: decoded.sub },
      include: { clientProfile: true },
    });

    if (!user) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'User not found',
        timestamp: new Date().toISOString(),
      });
    }

    req.user = user;
    next();
  } catch (err) {
    next(err);
  }
}

/**
 * requireRole - Factory that returns middleware enforcing role-based access.
 * Usage: requireRole('ADMIN') or requireRole('ADMIN', 'CLIENT')
 */
function requireRole(...roles) {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'Authentication required',
        timestamp: new Date().toISOString(),
      });
    }

    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        error: 'Forbidden',
        message: `Access denied. Required role: ${roles.join(' or ')}`,
        timestamp: new Date().toISOString(),
      });
    }

    next();
  };
}

module.exports = { verifyToken, requireRole };
