-- Community Features Tables for Aero Melody
-- Forums, Contests, Social Interactions
-- Uses FREE MariaDB features with JSON storage

USE aero_melody;

-- ============================================================================
-- COMMUNITY FEATURES TABLES
-- ============================================================================

-- 1. FORUM THREADS TABLE
CREATE TABLE IF NOT EXISTS forum_threads (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    tags JSON,
    views INT UNSIGNED DEFAULT 0,
    replies_count INT UNSIGNED DEFAULT 0,
    is_pinned TINYINT(1) DEFAULT 0,
    is_locked TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_forum_threads_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_forum_category (category),
    INDEX idx_forum_created (created_at DESC),
    INDEX idx_forum_pinned (is_pinned, created_at DESC),
    FULLTEXT INDEX idx_forum_search (title, content)
) ENGINE=InnoDB;

-- 2. FORUM REPLIES TABLE
CREATE TABLE IF NOT EXISTS forum_replies (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    thread_id INT UNSIGNED NOT NULL,
    user_id INT UNSIGNED NOT NULL,
    content TEXT NOT NULL,
    is_solution TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_forum_replies_thread FOREIGN KEY (thread_id)
        REFERENCES forum_threads(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_forum_replies_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_replies_thread (thread_id, created_at),
    INDEX idx_replies_user (user_id)
) ENGINE=InnoDB;

-- 3. CONTESTS TABLE
CREATE TABLE IF NOT EXISTS contests (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    creator_id INT UNSIGNED NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    rules JSON NOT NULL,
    prizes JSON,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    status ENUM('upcoming', 'active', 'ended', 'cancelled') DEFAULT 'upcoming',
    max_submissions INT UNSIGNED DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_contests_creator FOREIGN KEY (creator_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_contests_status (status, start_date),
    INDEX idx_contests_dates (start_date, end_date),
    FULLTEXT INDEX idx_contests_search (title, description)
) ENGINE=InnoDB;

-- 4. CONTEST SUBMISSIONS TABLE
CREATE TABLE IF NOT EXISTS contest_submissions (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    contest_id INT UNSIGNED NOT NULL,
    user_id INT UNSIGNED NOT NULL,
    composition_id INT UNSIGNED NOT NULL,
    description TEXT,
    votes_count INT UNSIGNED DEFAULT 0,
    rank INT UNSIGNED,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_submissions_contest FOREIGN KEY (contest_id)
        REFERENCES contests(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_submissions_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_submissions_composition FOREIGN KEY (composition_id)
        REFERENCES music_compositions(id) ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE KEY unique_contest_user (contest_id, user_id),
    INDEX idx_submissions_contest (contest_id, votes_count DESC),
    INDEX idx_submissions_user (user_id)
) ENGINE=InnoDB;

-- 5. CONTEST VOTES TABLE
CREATE TABLE IF NOT EXISTS contest_votes (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    submission_id INT UNSIGNED NOT NULL,
    user_id INT UNSIGNED NOT NULL,
    vote_value INT DEFAULT 1,
    voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_votes_submission FOREIGN KEY (submission_id)
        REFERENCES contest_submissions(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_votes_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE KEY unique_vote (submission_id, user_id),
    INDEX idx_votes_submission (submission_id)
) ENGINE=InnoDB;

-- 6. USER FOLLOWS TABLE
CREATE TABLE IF NOT EXISTS user_follows (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    follower_id INT UNSIGNED NOT NULL,
    following_id INT UNSIGNED NOT NULL,
    followed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_follows_follower FOREIGN KEY (follower_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_follows_following FOREIGN KEY (following_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE KEY unique_follow (follower_id, following_id),
    INDEX idx_follows_follower (follower_id),
    INDEX idx_follows_following (following_id)
) ENGINE=InnoDB;

-- 7. COMPOSITION LIKES TABLE
CREATE TABLE IF NOT EXISTS composition_likes (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    composition_id INT UNSIGNED NOT NULL,
    user_id INT UNSIGNED NOT NULL,
    liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_likes_composition FOREIGN KEY (composition_id)
        REFERENCES music_compositions(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_likes_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE KEY unique_like (composition_id, user_id),
    INDEX idx_likes_composition (composition_id),
    INDEX idx_likes_user (user_id),
    INDEX idx_likes_recent (liked_at DESC)
) ENGINE=InnoDB;

-- 8. COMPOSITION COMMENTS TABLE
CREATE TABLE IF NOT EXISTS composition_comments (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    composition_id INT UNSIGNED NOT NULL,
    user_id INT UNSIGNED NOT NULL,
    comment TEXT NOT NULL,
    parent_comment_id INT UNSIGNED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_comments_composition FOREIGN KEY (composition_id)
        REFERENCES music_compositions(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_comments_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_comments_parent FOREIGN KEY (parent_comment_id)
        REFERENCES composition_comments(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_comments_composition (composition_id, created_at DESC),
    INDEX idx_comments_user (user_id),
    FULLTEXT INDEX idx_comments_search (comment)
) ENGINE=InnoDB;

-- 9. USER ACHIEVEMENTS TABLE
CREATE TABLE IF NOT EXISTS user_achievements (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    achievement_type ENUM(
        'first_composition', 'ten_compositions', 'hundred_compositions',
        'first_remix', 'contest_winner', 'contest_participant',
        'popular_composer', 'community_contributor', 'early_adopter'
    ) NOT NULL,
    achievement_data JSON,
    earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_achievements_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE KEY unique_achievement (user_id, achievement_type),
    INDEX idx_achievements_user (user_id),
    INDEX idx_achievements_type (achievement_type)
) ENGINE=InnoDB;

-- 10. NOTIFICATIONS TABLE
CREATE TABLE IF NOT EXISTS notifications (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    notification_type ENUM(
        'new_follower', 'composition_liked', 'composition_commented',
        'contest_started', 'contest_ended', 'achievement_earned',
        'reply_received', 'mention'
    ) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT,
    related_id INT UNSIGNED,
    related_type VARCHAR(50),
    is_read TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_notifications_user FOREIGN KEY (user_id)
        REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    INDEX idx_notifications_user (user_id, is_read, created_at DESC),
    INDEX idx_notifications_type (notification_type)
) ENGINE=InnoDB;

-- ============================================================================
-- VIEWS FOR COMMUNITY FEATURES
-- ============================================================================

-- View: Popular Threads
DROP VIEW IF EXISTS popular_threads;
CREATE VIEW popular_threads AS
SELECT
    ft.id,
    ft.title,
    ft.category,
    ft.views,
    ft.replies_count,
    ft.created_at,
    u.username as author,
    COUNT(DISTINCT fr.id) as actual_replies
FROM forum_threads ft
JOIN users u ON ft.user_id = u.id
LEFT JOIN forum_replies fr ON ft.id = fr.thread_id
GROUP BY ft.id, ft.title, ft.category, ft.views, ft.replies_count, ft.created_at, u.username
ORDER BY ft.views DESC, ft.replies_count DESC;

-- View: Active Contests
DROP VIEW IF EXISTS active_contests_view;
CREATE VIEW active_contests_view AS
SELECT
    c.id,
    c.title,
    c.description,
    c.start_date,
    c.end_date,
    c.status,
    u.username as creator,
    COUNT(DISTINCT cs.id) as submissions_count,
    COUNT(DISTINCT cs.user_id) as participants_count
FROM contests c
JOIN users u ON c.creator_id = u.id
LEFT JOIN contest_submissions cs ON c.id = cs.contest_id
WHERE c.status IN ('upcoming', 'active')
GROUP BY c.id, c.title, c.description, c.start_date, c.end_date, c.status, u.username
ORDER BY c.start_date ASC;

-- View: Trending Compositions
DROP VIEW IF EXISTS trending_compositions_view;
CREATE VIEW trending_compositions_view AS
SELECT
    mc.id,
    mc.title,
    mc.genre,
    mc.tempo,
    mc.created_at,
    u.username as composer,
    COUNT(DISTINCT cl.id) as likes_count,
    COUNT(DISTINCT cc.id) as comments_count,
    (COUNT(DISTINCT cl.id) * 2 + COUNT(DISTINCT cc.id)) as trending_score
FROM music_compositions mc
JOIN users u ON mc.user_id = u.id
LEFT JOIN composition_likes cl ON mc.id = cl.composition_id
    AND cl.liked_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
LEFT JOIN composition_comments cc ON mc.id = cc.composition_id
    AND cc.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
WHERE mc.is_public = 1
GROUP BY mc.id, mc.title, mc.genre, mc.tempo, mc.created_at, u.username
HAVING trending_score > 0
ORDER BY trending_score DESC, mc.created_at DESC;

-- View: User Social Stats
DROP VIEW IF EXISTS user_social_stats;
CREATE VIEW user_social_stats AS
SELECT
    u.id as user_id,
    u.username,
    COUNT(DISTINCT uf1.id) as followers_count,
    COUNT(DISTINCT uf2.id) as following_count,
    COUNT(DISTINCT mc.id) as compositions_count,
    COUNT(DISTINCT cl.id) as total_likes_received,
    COUNT(DISTINCT cc.id) as total_comments_received
FROM users u
LEFT JOIN user_follows uf1 ON u.id = uf1.following_id
LEFT JOIN user_follows uf2 ON u.id = uf2.follower_id
LEFT JOIN music_compositions mc ON u.id = mc.user_id
LEFT JOIN composition_likes cl ON mc.id = cl.composition_id
LEFT JOIN composition_comments cc ON mc.id = cc.composition_id
GROUP BY u.id, u.username;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Additional indexes for trending and popular content
CREATE INDEX idx_likes_trending ON composition_likes(liked_at DESC, composition_id);
CREATE INDEX idx_comments_trending ON composition_comments(created_at DESC, composition_id);
CREATE INDEX idx_threads_popular ON forum_threads(views DESC, replies_count DESC);
CREATE INDEX idx_contests_active ON contests(status, start_date, end_date);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Trigger: Update replies count when reply is added
DROP TRIGGER IF EXISTS after_reply_insert;
DELIMITER //
CREATE TRIGGER after_reply_insert
AFTER INSERT ON forum_replies
FOR EACH ROW
BEGIN
    UPDATE forum_threads
    SET replies_count = replies_count + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.thread_id;
END//
DELIMITER ;

-- Trigger: Update votes count when vote is added
DROP TRIGGER IF EXISTS after_vote_insert;
DELIMITER //
CREATE TRIGGER after_vote_insert
AFTER INSERT ON contest_votes
FOR EACH ROW
BEGIN
    UPDATE contest_submissions
    SET votes_count = votes_count + NEW.vote_value
    WHERE id = NEW.submission_id;
END//
DELIMITER ;

-- Trigger: Update votes count when vote is deleted
DROP TRIGGER IF EXISTS after_vote_delete;
DELIMITER //
CREATE TRIGGER after_vote_delete
AFTER DELETE ON contest_votes
FOR EACH ROW
BEGIN
    UPDATE contest_submissions
    SET votes_count = votes_count - OLD.vote_value
    WHERE id = OLD.submission_id;
END//
DELIMITER ;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Show all new tables
SELECT TABLE_NAME, TABLE_ROWS, CREATE_TIME
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'aero_melody'
AND TABLE_NAME IN (
    'forum_threads', 'forum_replies', 'contests', 'contest_submissions',
    'contest_votes', 'user_follows', 'composition_likes', 'composition_comments',
    'user_achievements', 'notifications'
)
ORDER BY TABLE_NAME;

-- Show all views
SELECT TABLE_NAME
FROM information_schema.VIEWS
WHERE TABLE_SCHEMA = 'aero_melody'
ORDER BY TABLE_NAME;

SELECT 'Community features tables created successfully!' as status;
