"""
Real-time activity feed service for tracking and broadcasting user activities
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, desc, func

from app.models.models import UserActivity, ActivityType, User
from app.services.websocket_manager import WebSocketManager


@dataclass
class ActivityData:
    """Data structure for activity information"""
    user_id: int
    username: str
    activity_type: ActivityType
    target_id: Optional[int] = None
    target_type: Optional[str] = None
    activity_data: Optional[Dict] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class ActivityService:
    """Service for managing real-time user activity feed"""

    def __init__(self, websocket_manager: Optional[WebSocketManager] = None):
        self.websocket_manager = websocket_manager

    async def log_activity(
        self,
        db: AsyncSession,
        user_id: int,
        activity_type: ActivityType,
        target_id: Optional[int] = None,
        target_type: Optional[str] = None,
        activity_data: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Log a user activity and broadcast it in real-time"""
        try:
            # Get user information
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()

            if not user:
                return False

            # Create activity record
            activity = UserActivity(
                user_id=user_id,
                activity_type=activity_type,
                target_id=target_id,
                target_type=target_type,
                activity_data=activity_data,
                ip_address=ip_address,
                user_agent=user_agent
            )

            db.add(activity)
            await db.commit()
            await db.refresh(activity)

            # Broadcast activity in real-time
            await self._broadcast_activity(ActivityData(
                user_id=user_id,
                username=user.username,
                activity_type=activity_type,
                target_id=target_id,
                target_type=target_type,
                activity_data=activity_data,
                ip_address=ip_address,
                user_agent=user_agent
            ))

            return True
        except Exception as e:
            await db.rollback()
            raise e

    async def get_user_activities(
        self,
        db: AsyncSession,
        user_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
        activity_type: Optional[ActivityType] = None
    ) -> List[Dict]:
        """Get activities for a user or globally"""
        try:
            query = select(UserActivity).join(User).order_by(desc(UserActivity.created_at))

            if user_id:
                query = query.where(UserActivity.user_id == user_id)

            if activity_type:
                query = query.where(UserActivity.activity_type == activity_type)

            query = query.offset(offset).limit(limit)
            result = await db.execute(query)
            activities = result.scalars().all()

            return [
                {
                    "id": activity.id,
                    "user_id": activity.user_id,
                    "username": activity.user.username,
                    "activity_type": activity.activity_type.value,
                    "target_id": activity.target_id,
                    "target_type": activity.target_type,
                    "activity_data": activity.activity_data,
                    "created_at": activity.created_at.isoformat() if activity.created_at else None
                }
                for activity in activities
            ]
        except Exception as e:
            raise e

    async def get_recent_activities(
        self,
        db: AsyncSession,
        minutes: int = 60,
        limit: int = 100
    ) -> List[Dict]:
        """Get activities from the last N minutes"""
        try:
            since_time = datetime.utcnow() - timedelta(minutes=minutes)

            result = await db.execute(
                select(UserActivity)
                .join(User)
                .where(UserActivity.created_at >= since_time)
                .order_by(desc(UserActivity.created_at))
                .limit(limit)
            )
            activities = result.scalars().all()

            return [
                {
                    "id": activity.id,
                    "user_id": activity.user_id,
                    "username": activity.user.username,
                    "activity_type": activity.activity_type.value,
                    "target_id": activity.target_id,
                    "target_type": activity.target_type,
                    "activity_data": activity.activity_data,
                    "created_at": activity.created_at.isoformat() if activity.created_at else None
                }
                for activity in activities
            ]
        except Exception as e:
            raise e

    async def get_activity_stats(
        self,
        db: AsyncSession,
        user_id: Optional[int] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get activity statistics"""
        try:
            since_time = datetime.utcnow() - timedelta(days=days)

            query = select(
                UserActivity.activity_type,
                func.count(UserActivity.id).label('count')
            ).where(UserActivity.created_at >= since_time)

            if user_id:
                query = query.where(UserActivity.user_id == user_id)

            query = query.group_by(UserActivity.activity_type)
            result = await db.execute(query)
            stats = result.all()

            return {
                "total_activities": sum(stat.count for stat in stats),
                "activity_breakdown": {
                    stat.activity_type.value: stat.count for stat in stats
                },
                "period_days": days
            }
        except Exception as e:
            raise e

    async def cleanup_old_activities(
        self,
        db: AsyncSession,
        older_than_days: int = 90
    ) -> int:
        """Clean up old activities (older than specified days)"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

            result = await db.execute(
                select(UserActivity).where(UserActivity.created_at < cutoff_date)
            )
            old_activities = result.scalars().all()

            count = len(old_activities)
            for activity in old_activities:
                await db.delete(activity)

            await db.commit()
            return count
        except Exception as e:
            await db.rollback()
            raise e

    async def _broadcast_activity(self, activity_data: ActivityData):
        """Broadcast activity to all connected WebSocket clients"""
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast_activity({
                    "user_id": activity_data.user_id,
                    "username": activity_data.username,
                    "activity_type": activity_data.activity_type.value,
                    "target_id": activity_data.target_id,
                    "target_type": activity_data.target_type,
                    "activity_data": activity_data.activity_data,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                # Log error but don't fail the activity logging
                print(f"Failed to broadcast activity: {e}")

    async def track_composition_activity(
        self,
        db: AsyncSession,
        user_id: int,
        action: str,
        composition_id: int,
        composition_data: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Track composition-related activities"""
        activity_type = ActivityType.COMPOSITION_CREATED if action == "create" else \
                       ActivityType.COMPOSITION_UPDATED if action == "update" else \
                       ActivityType.COMPOSITION_DELETED if action == "delete" else None

        if not activity_type:
            return False

        return await self.log_activity(
            db=db,
            user_id=user_id,
            activity_type=activity_type,
            target_id=composition_id,
            target_type="composition",
            activity_data=composition_data,
            ip_address=ip_address,
            user_agent=user_agent
        )

    async def track_dataset_activity(
        self,
        db: AsyncSession,
        user_id: int,
        action: str,
        dataset_id: int,
        dataset_data: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Track dataset-related activities"""
        activity_type = ActivityType.DATASET_CREATED if action == "create" else \
                       ActivityType.DATASET_UPDATED if action == "update" else \
                       ActivityType.DATASET_DELETED if action == "delete" else None

        if not activity_type:
            return False

        return await self.log_activity(
            db=db,
            user_id=user_id,
            activity_type=activity_type,
            target_id=dataset_id,
            target_type="dataset",
            activity_data=dataset_data,
            ip_address=ip_address,
            user_agent=user_agent
        )

    async def track_collection_activity(
        self,
        db: AsyncSession,
        user_id: int,
        action: str,
        collection_id: int,
        collection_data: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Track collection-related activities"""
        activity_type = ActivityType.COLLECTION_CREATED if action == "create" else \
                       ActivityType.COLLECTION_UPDATED if action == "update" else \
                       ActivityType.COLLECTION_DELETED if action == "delete" else None

        if not activity_type:
            return False

        return await self.log_activity(
            db=db,
            user_id=user_id,
            activity_type=activity_type,
            target_id=collection_id,
            target_type="collection",
            activity_data=collection_data,
            ip_address=ip_address,
            user_agent=user_agent
        )

    async def track_collaboration_activity(
        self,
        db: AsyncSession,
        user_id: int,
        action: str,
        session_id: Optional[int] = None,
        session_data: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Track collaboration-related activities"""
        activity_type = ActivityType.COLLABORATION_JOINED if action == "join" else \
                       ActivityType.COLLABORATION_LEFT if action == "leave" else \
                       ActivityType.COLLABORATION_UPDATED if action == "update" else None

        if not activity_type:
            return False

        return await self.log_activity(
            db=db,
            user_id=user_id,
            activity_type=activity_type,
            target_id=session_id,
            target_type="collaboration",
            activity_data=session_data,
            ip_address=ip_address,
            user_agent=user_agent
        )

    async def track_search_activity(
        self,
        db: AsyncSession,
        user_id: int,
        search_query: str,
        search_type: str = "vector",
        results_count: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Track search activities"""
        return await self.log_activity(
            db=db,
            user_id=user_id,
            activity_type=ActivityType.SEARCH_PERFORMED,
            activity_data={
                "query": search_query,
                "search_type": search_type,
                "results_count": results_count
            },
            ip_address=ip_address,
            user_agent=user_agent
        )
