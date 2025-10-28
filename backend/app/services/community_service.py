"""
Community features service for forums, contests, and social interactions
Real-time features using FREE MariaDB capabilities
"""

from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, or_, func, desc
from datetime import datetime, timedelta
import json
import logging

from app.models.models import User, MusicComposition

logger = logging.getLogger(__name__)


class ForumService:
    """Forum and discussion management"""

    async def create_thread(
        self,
        db: AsyncSession,
        user_id: int,
        title: str,
        content: str,
        category: str,
        tags: Optional[List[str]] = None
    ) -> Dict:
        """Create a new forum thread"""
        try:
            thread_data = {
                'user_id': user_id,
                'title': title,
                'content': content,
                'category': category,
                'tags': json.dumps(tags or []),
                'created_at': datetime.utcnow(),
                'views': 0,
                'replies_count': 0,
                'is_pinned': False,
                'is_locked': False
            }

            # Store in user_activities as forum_thread_created
            from app.models.models import UserActivity, ActivityType
            activity = UserActivity(
                user_id=user_id,
                activity_type=ActivityType.COMPOSITION_CREATED,  # Reuse for now
                target_type='forum_thread',
                activity_data=thread_data
            )
            db.add(activity)
            await db.commit()
            await db.refresh(activity)

            return {
                'thread_id': activity.id,
                'title': title,
                'created_at': activity.created_at.isoformat(),
                'message': 'Thread created successfully'
            }

        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            await db.rollback()
            raise

    async def get_threads(
        self,
        db: AsyncSession,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Get forum threads"""
        try:
            from app.models.models import UserActivity

            query = select(UserActivity).where(
                UserActivity.target_type == 'forum_thread'
            )

            if category:
                # Filter by category in JSON data
                query = query.where(
                    func.json_extract(UserActivity.activity_data, '$.category') == category
                )

            query = query.order_by(desc(UserActivity.created_at)).limit(limit).offset(offset)

            result = await db.execute(query)
            activities = result.scalars().all()

            threads = []
            for activity in activities:
                thread_data = activity.activity_data or {}
                threads.append({
                    'thread_id': activity.id,
                    'user_id': activity.user_id,
                    'title': thread_data.get('title'),
                    'category': thread_data.get('category'),
                    'tags': json.loads(thread_data.get('tags', '[]')),
                    'views': thread_data.get('views', 0),
                    'replies_count': thread_data.get('replies_count', 0),
                    'created_at': activity.created_at.isoformat()
                })

            return threads

        except Exception as e:
            logger.error(f"Error getting threads: {e}")
            return []

    async def post_reply(
        self,
        db: AsyncSession,
        thread_id: int,
        user_id: int,
        content: str
    ) -> Dict:
        """Post a reply to a thread"""
        try:
            from app.models.models import UserActivity, ActivityType

            reply_data = {
                'thread_id': thread_id,
                'user_id': user_id,
                'content': content,
                'created_at': datetime.utcnow().isoformat()
            }

            activity = UserActivity(
                user_id=user_id,
                activity_type=ActivityType.COMPOSITION_UPDATED,  # Reuse for now
                target_id=thread_id,
                target_type='forum_reply',
                activity_data=reply_data
            )
            db.add(activity)
            await db.commit()

            return {
                'reply_id': activity.id,
                'thread_id': thread_id,
                'message': 'Reply posted successfully'
            }

        except Exception as e:
            logger.error(f"Error posting reply: {e}")
            await db.rollback()
            raise


class ContestService:
    """Contest and competition management"""

    async def create_contest(
        self,
        db: AsyncSession,
        creator_id: int,
        title: str,
        description: str,
        start_date: datetime,
        end_date: datetime,
        rules: Dict,
        prizes: Optional[List[Dict]] = None
    ) -> Dict:
        """Create a new composition contest"""
        try:
            from app.models.models import UserActivity, ActivityType

            contest_data = {
                'creator_id': creator_id,
                'title': title,
                'description': description,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'rules': rules,
                'prizes': prizes or [],
                'status': 'upcoming',
                'participants': [],
                'submissions': []
            }

            activity = UserActivity(
                user_id=creator_id,
                activity_type=ActivityType.COLLECTION_CREATED,  # Reuse for now
                target_type='contest',
                activity_data=contest_data
            )
            db.add(activity)
            await db.commit()
            await db.refresh(activity)

            return {
                'contest_id': activity.id,
                'title': title,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'message': 'Contest created successfully'
            }

        except Exception as e:
            logger.error(f"Error creating contest: {e}")
            await db.rollback()
            raise

    async def get_active_contests(
        self,
        db: AsyncSession
    ) -> List[Dict]:
        """Get currently active contests"""
        try:
            from app.models.models import UserActivity

            result = await db.execute(
                select(UserActivity).where(
                    UserActivity.target_type == 'contest'
                ).order_by(desc(UserActivity.created_at))
            )
            activities = result.scalars().all()

            contests = []
            now = datetime.utcnow()

            for activity in activities:
                contest_data = activity.activity_data or {}
                start_date = datetime.fromisoformat(contest_data.get('start_date', ''))
                end_date = datetime.fromisoformat(contest_data.get('end_date', ''))

                if start_date <= now <= end_date:
                    contests.append({
                        'contest_id': activity.id,
                        'title': contest_data.get('title'),
                        'description': contest_data.get('description'),
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'participants_count': len(contest_data.get('participants', [])),
                        'submissions_count': len(contest_data.get('submissions', []))
                    })

            return contests

        except Exception as e:
            logger.error(f"Error getting contests: {e}")
            return []

    async def submit_to_contest(
        self,
        db: AsyncSession,
        contest_id: int,
        user_id: int,
        composition_id: int,
        description: Optional[str] = None
    ) -> Dict:
        """Submit a composition to a contest"""
        try:
            from app.models.models import UserActivity

            # Get contest
            result = await db.execute(
                select(UserActivity).where(UserActivity.id == contest_id)
            )
            contest_activity = result.scalar_one_or_none()

            if not contest_activity:
                raise ValueError("Contest not found")

            # Add submission to contest data
            contest_data = contest_activity.activity_data or {}
            submissions = contest_data.get('submissions', [])
            
            submission = {
                'user_id': user_id,
                'composition_id': composition_id,
                'description': description,
                'submitted_at': datetime.utcnow().isoformat(),
                'votes': 0
            }
            submissions.append(submission)
            contest_data['submissions'] = submissions

            # Update participants
            participants = contest_data.get('participants', [])
            if user_id not in participants:
                participants.append(user_id)
                contest_data['participants'] = participants

            contest_activity.activity_data = contest_data
            await db.commit()

            return {
                'message': 'Submission successful',
                'contest_id': contest_id,
                'composition_id': composition_id
            }

        except Exception as e:
            logger.error(f"Error submitting to contest: {e}")
            await db.rollback()
            raise

    async def vote_for_submission(
        self,
        db: AsyncSession,
        contest_id: int,
        composition_id: int,
        user_id: int
    ) -> Dict:
        """Vote for a contest submission"""
        try:
            from app.models.models import UserActivity

            result = await db.execute(
                select(UserActivity).where(UserActivity.id == contest_id)
            )
            contest_activity = result.scalar_one_or_none()

            if not contest_activity:
                raise ValueError("Contest not found")

            contest_data = contest_activity.activity_data or {}
            submissions = contest_data.get('submissions', [])

            # Find and update submission votes
            for submission in submissions:
                if submission['composition_id'] == composition_id:
                    submission['votes'] = submission.get('votes', 0) + 1
                    break

            contest_data['submissions'] = submissions
            contest_activity.activity_data = contest_data
            await db.commit()

            return {'message': 'Vote recorded successfully'}

        except Exception as e:
            logger.error(f"Error voting: {e}")
            await db.rollback()
            raise

    async def get_contest_leaderboard(
        self,
        db: AsyncSession,
        contest_id: int
    ) -> List[Dict]:
        """Get contest leaderboard"""
        try:
            from app.models.models import UserActivity

            result = await db.execute(
                select(UserActivity).where(UserActivity.id == contest_id)
            )
            contest_activity = result.scalar_one_or_none()

            if not contest_activity:
                return []

            contest_data = contest_activity.activity_data or {}
            submissions = contest_data.get('submissions', [])

            # Sort by votes
            leaderboard = sorted(submissions, key=lambda x: x.get('votes', 0), reverse=True)

            return leaderboard

        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []


class SocialInteractionService:
    """Social features like follows, likes, comments"""

    async def follow_user(
        self,
        db: AsyncSession,
        follower_id: int,
        following_id: int
    ) -> Dict:
        """Follow another user"""
        try:
            from app.models.models import UserActivity, ActivityType

            follow_data = {
                'follower_id': follower_id,
                'following_id': following_id,
                'followed_at': datetime.utcnow().isoformat()
            }

            activity = UserActivity(
                user_id=follower_id,
                activity_type=ActivityType.COLLABORATION_JOINED,  # Reuse for now
                target_id=following_id,
                target_type='user_follow',
                activity_data=follow_data
            )
            db.add(activity)
            await db.commit()

            return {'message': 'User followed successfully'}

        except Exception as e:
            logger.error(f"Error following user: {e}")
            await db.rollback()
            raise

    async def like_composition(
        self,
        db: AsyncSession,
        user_id: int,
        composition_id: int
    ) -> Dict:
        """Like a composition"""
        try:
            from app.models.models import UserActivity, ActivityType

            like_data = {
                'user_id': user_id,
                'composition_id': composition_id,
                'liked_at': datetime.utcnow().isoformat()
            }

            activity = UserActivity(
                user_id=user_id,
                activity_type=ActivityType.COMPOSITION_UPDATED,  # Reuse for now
                target_id=composition_id,
                target_type='composition_like',
                activity_data=like_data
            )
            db.add(activity)
            await db.commit()

            return {'message': 'Composition liked successfully'}

        except Exception as e:
            logger.error(f"Error liking composition: {e}")
            await db.rollback()
            raise

    async def comment_on_composition(
        self,
        db: AsyncSession,
        user_id: int,
        composition_id: int,
        comment: str
    ) -> Dict:
        """Comment on a composition"""
        try:
            from app.models.models import UserActivity, ActivityType

            comment_data = {
                'user_id': user_id,
                'composition_id': composition_id,
                'comment': comment,
                'commented_at': datetime.utcnow().isoformat()
            }

            activity = UserActivity(
                user_id=user_id,
                activity_type=ActivityType.COMPOSITION_UPDATED,  # Reuse for now
                target_id=composition_id,
                target_type='composition_comment',
                activity_data=comment_data
            )
            db.add(activity)
            await db.commit()

            return {
                'comment_id': activity.id,
                'message': 'Comment posted successfully'
            }

        except Exception as e:
            logger.error(f"Error posting comment: {e}")
            await db.rollback()
            raise

    async def get_composition_comments(
        self,
        db: AsyncSession,
        composition_id: int,
        limit: int = 50
    ) -> List[Dict]:
        """Get comments for a composition"""
        try:
            from app.models.models import UserActivity

            result = await db.execute(
                select(UserActivity).where(
                    and_(
                        UserActivity.target_id == composition_id,
                        UserActivity.target_type == 'composition_comment'
                    )
                ).order_by(desc(UserActivity.created_at)).limit(limit)
            )
            activities = result.scalars().all()

            comments = []
            for activity in activities:
                comment_data = activity.activity_data or {}
                comments.append({
                    'comment_id': activity.id,
                    'user_id': activity.user_id,
                    'comment': comment_data.get('comment'),
                    'created_at': activity.created_at.isoformat()
                })

            return comments

        except Exception as e:
            logger.error(f"Error getting comments: {e}")
            return []

    async def get_trending_compositions(
        self,
        db: AsyncSession,
        days: int = 7,
        limit: int = 20
    ) -> List[Dict]:
        """Get trending compositions based on likes and comments"""
        try:
            from app.models.models import UserActivity

            since_date = datetime.utcnow() - timedelta(days=days)

            # Get compositions with most interactions
            result = await db.execute(
                select(
                    UserActivity.target_id,
                    func.count(UserActivity.id).label('interaction_count')
                ).where(
                    and_(
                        UserActivity.target_type.in_(['composition_like', 'composition_comment']),
                        UserActivity.created_at >= since_date
                    )
                ).group_by(UserActivity.target_id)
                .order_by(desc('interaction_count'))
                .limit(limit)
            )

            trending_data = result.all()

            trending = []
            for composition_id, count in trending_data:
                # Get composition details
                comp_result = await db.execute(
                    select(MusicComposition).where(MusicComposition.id == composition_id)
                )
                composition = comp_result.scalar_one_or_none()

                if composition:
                    trending.append({
                        'composition_id': composition.id,
                        'title': composition.title,
                        'genre': composition.genre,
                        'interaction_count': count,
                        'created_at': composition.created_at.isoformat()
                    })

            return trending

        except Exception as e:
            logger.error(f"Error getting trending compositions: {e}")
            return []


class CommunityManager:
    """Main community features manager"""

    def __init__(self):
        self.forum_service = ForumService()
        self.contest_service = ContestService()
        self.social_service = SocialInteractionService()

    async def get_community_stats(self, db: AsyncSession) -> Dict:
        """Get overall community statistics"""
        try:
            from app.models.models import User, MusicComposition, UserActivity

            # Count users
            users_result = await db.execute(select(func.count(User.id)))
            total_users = users_result.scalar()

            # Count compositions
            comps_result = await db.execute(select(func.count(MusicComposition.id)))
            total_compositions = comps_result.scalar()

            # Count activities
            activities_result = await db.execute(select(func.count(UserActivity.id)))
            total_activities = activities_result.scalar()

            # Get active contests
            active_contests = await self.contest_service.get_active_contests(db)

            return {
                'total_users': total_users,
                'total_compositions': total_compositions,
                'total_activities': total_activities,
                'active_contests': len(active_contests),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting community stats: {e}")
            return {}
