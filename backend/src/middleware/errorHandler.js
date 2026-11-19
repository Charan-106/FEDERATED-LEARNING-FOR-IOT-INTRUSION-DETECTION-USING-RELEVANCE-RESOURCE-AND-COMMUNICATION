// Central error handler for ARC-FL Backend Express app
// Must be registered as the last middleware (after all routes)

/**
 * errorHandler - Catches all errors passed via next(err) and returns
 * a structured JSON response. Prevents stack traces from leaking in production.
 */
function errorHandler(err, req, res, next) {
  // Log the full error server-side
  console.error(`[ErrorHandler] ${err.message}`, {
    path: req.path,
    method: req.method,
    stack: err.stack,
  });

  // Determine HTTP status code
  const statusCode = err.statusCode || err.status || 500;

  // Build structured response
  const response = {
    error: err.name || 'InternalServerError',
    message: statusCode < 500
      ? err.message
      : 'An unexpected error occurred. Please try again later.',
    timestamp: new Date().toISOString(),
  };

  // Include validation errors if present (from express-validator)
  if (err.errors) {
    response.errors = err.errors;
  }

  res.status(statusCode).json(response);
}

module.exports = errorHandler;
