"""
User dataset and collection management services
"""

from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_
import json

from app.models.models import (
    UserDataset, UserCollection, MusicComposition,
    CompositionRemix, RemixType, User, ActivityType
)
from app.services.activity_service import ActivityService


class UserDatasetManager:
    """Manages user personal collections and datasets"""

    async def create_dataset(
        self,
        db: AsyncSession,
        user_id: int,
        name: str,
        route_data: Dict,
        metadata: Optional[Dict] = None
    ) -> UserDataset:
        """Create a new user dataset"""
        dataset = UserDataset(
            user_id=user_id,
            name=name,
            route_data=route_data,
            metadata=metadata or {}
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)

        # Log activity
        activity_service = ActivityService()
        await activity_service.track_dataset_activity(
            db=db,
            user_id=user_id,
            action="create",
            dataset_id=dataset.id,
            dataset_data={"name": name, "route_count": len(route_data.get("routes", []))}
        )

        return dataset

    async def get_dataset(
        self,
        db: AsyncSession,
        dataset_id: int,
        user_id: int
    ) -> Optional[UserDataset]:
        """Get a specific dataset for a user"""
        result = await db.execute(
            select(UserDataset).where(
                and_(
                    UserDataset.id == dataset_id,
                    UserDataset.user_id == user_id
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_user_datasets(
        self,
        db: AsyncSession,
        user_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserDataset]:
        """Get all datasets for a user"""
        result = await db.execute(
            select(UserDataset)
            .where(UserDataset.user_id == user_id)
            .order_by(UserDataset.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def update_dataset(
        self,
        db: AsyncSession,
        dataset_id: int,
        user_id: int,
        name: Optional[str] = None,
        route_data: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[UserDataset]:
        """Update a dataset"""
        dataset = await self.get_dataset(db, dataset_id, user_id)
        if not dataset:
            return None

        if name:
            dataset.name = name
        if route_data:
            dataset.route_data = route_data
        if metadata:
            dataset.metadata = metadata

        await db.commit()
        await db.refresh(dataset)

        # Log activity
        activity_service = ActivityService()
        await activity_service.track_dataset_activity(
            db=db,
            user_id=user_id,
            action="update",
            dataset_id=dataset_id,
            dataset_data={"name": dataset.name}
        )

        return dataset

    async def delete_dataset(
        self,
        db: AsyncSession,
        dataset_id: int,
        user_id: int
    ) -> bool:
        """Delete a dataset and associated compositions"""
        dataset = await self.get_dataset(db, dataset_id, user_id)
        if not dataset:
            return False

        # Log activity before deletion
        activity_service = ActivityService()
        await activity_service.track_dataset_activity(
            db=db,
            user_id=user_id,
            action="delete",
            dataset_id=dataset_id,
            dataset_data={"name": dataset.name}
        )

        await db.delete(dataset)
        await db.commit()
        return True

    async def get_dataset_compositions(
        self,
        db: AsyncSession,
        dataset_id: int,
        user_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[MusicComposition]:
        """Get all compositions in a dataset"""
        dataset = await self.get_dataset(db, dataset_id, user_id)
        if not dataset:
            return []

        result = await db.execute(
            select(MusicComposition)
            .where(MusicComposition.dataset_id == dataset_id)
            .order_by(MusicComposition.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def add_composition_to_dataset(
        self,
        db: AsyncSession,
        dataset_id: int,
        composition_id: int,
        user_id: int
    ) -> bool:
        """Add a composition to a dataset"""
        dataset = await self.get_dataset(db, dataset_id, user_id)
        if not dataset:
            return False

        result = await db.execute(
            select(MusicComposition).where(MusicComposition.id == composition_id)
        )
        composition = result.scalar_one_or_none()

        if composition:
            composition.dataset_id = dataset_id
            await db.commit()
            return True

        return False


class CollectionManager:
    """Manages user composition collections"""

    async def create_collection(
        self,
        db: AsyncSession,
        user_id: int,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> UserCollection:
        """Create a new collection"""
        collection = UserCollection(
            user_id=user_id,
            name=name,
            description=description,
            composition_ids=[],
            tags=tags or []
        )
        db.add(collection)
        await db.commit()
        await db.refresh(collection)

        # Log activity
        activity_service = ActivityService()
        await activity_service.track_collection_activity(
            db=db,
            user_id=user_id,
            action="create",
            collection_id=collection.id,
            collection_data={"name": name, "tag_count": len(tags or [])}
        )

        return collection

    async def get_collection(
        self,
        db: AsyncSession,
        collection_id: int,
        user_id: int
    ) -> Optional[UserCollection]:
        """Get a specific collection"""
        result = await db.execute(
            select(UserCollection).where(
                and_(
                    UserCollection.id == collection_id,
                    UserCollection.user_id == user_id
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_user_collections(
        self,
        db: AsyncSession,
        user_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserCollection]:
        """Get all collections for a user"""
        result = await db.execute(
            select(UserCollection)
            .where(UserCollection.user_id == user_id)
            .order_by(UserCollection.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def add_composition_to_collection(
        self,
        db: AsyncSession,
        collection_id: int,
        composition_id: int,
        user_id: int
    ) -> bool:
        """Add a composition to a collection"""
        collection = await self.get_collection(db, collection_id, user_id)
        if not collection:
            return False

        if not collection.composition_ids:
            collection.composition_ids = []

        if composition_id not in collection.composition_ids:
            collection.composition_ids.append(composition_id)
            await db.commit()

        return True

    async def remove_composition_from_collection(
        self,
        db: AsyncSession,
        collection_id: int,
        composition_id: int,
        user_id: int
    ) -> bool:
        """Remove a composition from a collection"""
        collection = await self.get_collection(db, collection_id, user_id)
        if not collection:
            return False

        if collection.composition_ids and composition_id in collection.composition_ids:
            collection.composition_ids.remove(composition_id)
            await db.commit()

        return True

    async def update_collection(
        self,
        db: AsyncSession,
        collection_id: int,
        user_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[UserCollection]:
        """Update collection metadata"""
        collection = await self.get_collection(db, collection_id, user_id)
        if not collection:
            return None

        if name:
            collection.name = name
        if description:
            collection.description = description
        if tags is not None:
            collection.tags = tags

        await db.commit()
        await db.refresh(collection)

        # Log activity
        activity_service = ActivityService()
        await activity_service.track_collection_activity(
            db=db,
            user_id=user_id,
            action="update",
            collection_id=collection_id,
            collection_data={"name": collection.name}
        )

        return collection

    async def delete_collection(
        self,
        db: AsyncSession,
        collection_id: int,
        user_id: int
    ) -> bool:
        """Delete a collection"""
        collection = await self.get_collection(db, collection_id, user_id)
        if not collection:
            return False

        # Log activity before deletion
        activity_service = ActivityService()
        await activity_service.track_collection_activity(
            db=db,
            user_id=user_id,
            action="delete",
            collection_id=collection_id,
            collection_data={"name": collection.name}
        )

        await db.delete(collection)
        await db.commit()
        return True

    async def get_collection_compositions(
        self,
        db: AsyncSession,
        collection_id: int,
        user_id: int
    ) -> List[MusicComposition]:
        """Get all compositions in a collection"""
        collection = await self.get_collection(db, collection_id, user_id)
        if not collection or not collection.composition_ids:
            return []

        result = await db.execute(
            select(MusicComposition).where(
                MusicComposition.id.in_(collection.composition_ids)
            )
        )
        return result.scalars().all()

    async def search_collections_by_tag(
        self,
        db: AsyncSession,
        user_id: int,
        tag: str
    ) -> List[UserCollection]:
        """Search collections by tag"""
        result = await db.execute(
            select(UserCollection).where(
                UserCollection.user_id == user_id
            )
        )
        collections = result.scalars().all()

        # Filter collections that contain the tag
        matching = [
            c for c in collections
            if c.tags and tag.lower() in [t.lower() for t in c.tags]
        ]
        return matching


class RemixManager:
    """Manages composition remixes and attribution"""

    async def create_remix(
        self,
        db: AsyncSession,
        original_composition_id: int,
        remix_composition_id: int,
        remix_type: RemixType,
        attribution_data: Optional[Dict] = None
    ) -> CompositionRemix:
        """Create a remix relationship"""
        remix = CompositionRemix(
            original_composition_id=original_composition_id,
            remix_composition_id=remix_composition_id,
            remix_type=remix_type,
            attribution_data=attribution_data or {}
        )
        db.add(remix)
        await db.commit()
        await db.refresh(remix)

        # Get user_id from the remix composition
        result = await db.execute(
            select(MusicComposition).where(MusicComposition.id == remix_composition_id)
        )
        remix_composition = result.scalar_one_or_none()

        if remix_composition and remix_composition.user_id:
            # Log activity
            activity_service = ActivityService()
            await activity_service.log_activity(
                db=db,
                user_id=remix_composition.user_id,
                activity_type=ActivityType.REMIX_CREATED,
                target_id=remix_composition_id,
                target_type="composition",
                activity_data={
                    "original_id": original_composition_id,
                    "remix_type": remix_type.value,
                    "attribution": attribution_data
                }
            )

        return remix

    async def get_remix_chain(
        self,
        db: AsyncSession,
        composition_id: int
    ) -> Dict:
        """Get remix chain for a composition"""
        result = await db.execute(
            select(CompositionRemix).where(
                CompositionRemix.original_composition_id == composition_id
            )
        )
        remixes = result.scalars().all()

        chain = {
            "original_id": composition_id,
            "remixes": []
        }

        for remix in remixes:
            chain["remixes"].append({
                "remix_id": remix.remix_composition_id,
                "type": remix.remix_type.value,
                "created_at": remix.created_at.isoformat(),
                "attribution": remix.attribution_data
            })

        return chain

    async def get_remix_history(
        self,
        db: AsyncSession,
        composition_id: int,
        depth: int = 5
    ) -> List[Dict]:
        """Get full remix history for a composition"""
        history = []
        visited = set()

        async def traverse_remixes(comp_id: int, level: int):
            if level > depth or comp_id in visited:
                return

            visited.add(comp_id)

            result = await db.execute(
                select(MusicComposition).where(MusicComposition.id == comp_id)
            )
            composition = result.scalar_one_or_none()

            if composition:
                history.append({
                    "composition_id": comp_id,
                    "title": composition.title,
                    "level": level,
                    "created_at": composition.created_at.isoformat()
                })

                # Get remixes of this composition
                remix_result = await db.execute(
                    select(CompositionRemix).where(
                        CompositionRemix.original_composition_id == comp_id
                    )
                )
                remixes = remix_result.scalars().all()

                for remix in remixes:
                    await traverse_remixes(remix.remix_composition_id, level + 1)

        await traverse_remixes(composition_id, 0)
        return history

    async def get_original_composition(
        self,
        db: AsyncSession,
        remix_id: int
    ) -> Optional[MusicComposition]:
        """Get the original composition for a remix"""
        result = await db.execute(
            select(CompositionRemix).where(
                CompositionRemix.remix_composition_id == remix_id
            )
        )
        remix = result.scalar_one_or_none()

        if remix:
            original_result = await db.execute(
                select(MusicComposition).where(
                    MusicComposition.id == remix.original_composition_id
                )
            )
            return original_result.scalar_one_or_none()

        return None

    async def delete_remix(
        self,
        db: AsyncSession,
        remix_id: int
    ) -> bool:
        """Delete a remix relationship"""
        result = await db.execute(
            select(CompositionRemix).where(CompositionRemix.id == remix_id)
        )
        remix = result.scalar_one_or_none()

        if remix:
            await db.delete(remix)
            await db.commit()
            return True

        return False
