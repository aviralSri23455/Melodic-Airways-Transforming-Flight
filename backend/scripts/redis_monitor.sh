#!/bin/bash
# Redis Storage Monitor for Aero Melody
# Run this script every 15 minutes via cron to monitor storage usage

# Configuration
REDIS_URL="redis://default:zcUJQD3G4uebZD0Ve5hz6J171zwohat2@redis-16441.c267.us-east-1-4.ec2.redns.redis-cloud.com:16441"
LOG_FILE="/var/log/aero_melody/redis_monitor.log"
ALERT_THRESHOLD=25  # MB
CRITICAL_THRESHOLD=28  # MB

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Get current storage usage
STORAGE_INFO=$(python3 -c "
import redis
import json
from datetime import datetime

try:
    r = redis.from_url('$REDIS_URL', decode_responses=True)
    info = r.info()
    memory_used = info.get('used_memory', 0) / (1024 * 1024)  # MB
    memory_limit = 30

    result = {
        'timestamp': datetime.utcnow().isoformat(),
        'memory_used_mb': round(memory_used, 2),
        'memory_limit_mb': memory_limit,
        'memory_usage_percent': round((memory_used / memory_limit) * 100, 1),
        'status': 'CRITICAL' if memory_used > $CRITICAL_THRESHOLD else 'WARNING' if memory_used > $ALERT_THRESHOLD else 'OK',
        'total_keys': info.get('db0', {}).get('keys', 0),
        'connected_clients': info.get('connected_clients', 0)
    }

    print(json.dumps(result))
except Exception as e:
    print(json.dumps({'error': str(e), 'timestamp': datetime.utcnow().isoformat()}))
")

# Log the result
echo "$(date '+%Y-%m-%d %H:%M:%S') - $STORAGE_INFO" >> "$LOG_FILE"

# Parse the JSON result
MEMORY_USED=$(echo "$STORAGE_INFO" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'error' not in data:
    print(data['memory_used_mb'])
else:
    print('ERROR')
")

STATUS=$(echo "$STORAGE_INFO" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'error' not in data:
    print(data['status'])
else:
    print('ERROR')
")

# Send alerts if needed
if [ "$STATUS" = "WARNING" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Redis storage usage is $MEMORY_USED MB (over $ALERT_THRESHOLD MB threshold)" | tee -a "$LOG_FILE"

    # Optional: Send email alert
    # echo "Redis storage usage warning: $MEMORY_USED MB used" | mail -s "Redis Storage Warning" admin@yourdomain.com

elif [ "$STATUS" = "CRITICAL" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - CRITICAL: Redis storage usage is $MEMORY_USED MB (over $CRITICAL_THRESHOLD MB threshold)" | tee -a "$LOG_FILE"

    # Optional: Send critical email alert
    # echo "Redis storage usage CRITICAL: $MEMORY_USED MB used" | mail -s "Redis Storage CRITICAL" admin@yourdomain.com

    # Run cleanup script
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Running automatic cleanup..." | tee -a "$LOG_FILE"
    python3 scripts/redis_cleanup.py --dry-run

elif [ "$STATUS" = "OK" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - OK: Redis storage usage is $MEMORY_USED MB" >> "$LOG_FILE"
fi

# Keep log file manageable (keep last 1000 lines)
tail -n 1000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
