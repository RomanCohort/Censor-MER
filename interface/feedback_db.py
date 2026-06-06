# =============================================================================
# Feedback Database Layer - SQLite Storage for Human Feedback
# =============================================================================
# 替代JSON存储，支持:
#   - SQLite数据库CRUD操作
#   - 多格式导出 (JSON/CSV/Excel)
#   - 数据清洗和统计查询
#   - 提示词模板管理
# =============================================================================

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd

# =============================================================================
# Database Schema
# =============================================================================

SCHEMA_SQL = """
-- 主反馈表
CREATE TABLE IF NOT EXISTS feedbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT DEFAULT 'anonymous',
    image_path TEXT,
    prompt TEXT NOT NULL,
    video_path TEXT,

    -- 核心评分 (1-5分)
    naturalness REAL DEFAULT 3.0,
    smoothness REAL DEFAULT 3.0,
    prompt_match REAL DEFAULT 3.0,
    overall REAL DEFAULT 3.0,

    -- 专家级评分 (可选)
    au_accuracy REAL,
    micro_quality REAL,
    expert_comments TEXT,
    user_comments TEXT,

    -- 生成信息
    emotion TEXT,
    intensity REAL,
    active_au TEXT,

    -- 元数据
    is_comparison INTEGER DEFAULT 0,
    comparison_winner TEXT,
    is_deleted INTEGER DEFAULT 0,
    quality_flag TEXT DEFAULT 'normal'  -- normal, low, high
);

-- 提示词模板表
CREATE TABLE IF NOT EXISTS prompt_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    template TEXT NOT NULL,
    category TEXT DEFAULT 'basic',  -- basic, complex, sequence
    emotion TEXT,
    intensity TEXT,
    usage_count INTEGER DEFAULT 0,
    is_custom INTEGER DEFAULT 0,
    created_at TEXT
);

-- 批量任务表
CREATE TABLE IF NOT EXISTS batch_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT,
    created_at TEXT,
    status TEXT DEFAULT 'pending',  -- pending, running, completed
    total_items INTEGER,
    completed_items INTEGER DEFAULT 0
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_feedbacks_timestamp ON feedbacks(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedbacks_prompt ON feedbacks(prompt);
CREATE INDEX IF NOT EXISTS idx_feedbacks_overall ON feedbacks(overall);
CREATE INDEX IF NOT EXISTS idx_prompt_templates_category ON prompt_templates(category);
"""

# =============================================================================
# Default Prompt Templates
# =============================================================================

DEFAULT_PROMPT_TEMPLATES = [
    # Basic - 单一情感
    {'name': '微笑', 'template': '微笑', 'category': 'basic', 'emotion': 'happiness', 'intensity': 'medium'},
    {'name': '惊讶', 'template': '惊讶', 'category': 'basic', 'emotion': 'surprise', 'intensity': 'medium'},
    {'name': '厌恶', 'template': '厌恶', 'category': 'basic', 'emotion': 'disgust', 'intensity': 'medium'},
    {'name': '恐惧', 'template': '恐惧', 'category': 'basic', 'emotion': 'fear', 'intensity': 'medium'},
    {'name': '愤怒', 'template': '愤怒', 'category': 'basic', 'emotion': 'anger', 'intensity': 'medium'},
    {'name': '悲伤', 'template': '悲伤', 'category': 'basic', 'emotion': 'repression', 'intensity': 'medium'},
    {'name': ' contempt', 'template': ' contempt', 'category': 'basic', 'emotion': 'contempt', 'intensity': 'medium'},  # 轻蔑

    # Intensity variations - 强度变化
    {'name': '轻微微笑', 'template': '轻微微笑', 'category': 'intensity', 'emotion': 'happiness', 'intensity': 'weak'},
    {'name': '强烈微笑', 'template': '强烈微笑', 'category': 'intensity', 'emotion': 'happiness', 'intensity': 'strong'},
    {'name': '轻微惊讶', 'template': '轻微惊讶', 'category': 'intensity', 'emotion': 'surprise', 'intensity': 'weak'},
    {'name': '强烈惊讶', 'template': '强烈惊讶', 'category': 'intensity', 'emotion': 'surprise', 'intensity': 'strong'},
    {'name': '轻微厌恶', 'template': '轻微厌恶', 'category': 'intensity', 'emotion': 'disgust', 'intensity': 'weak'},
    {'name': '强烈厌恶', 'template': '强烈厌恶', 'category': 'intensity', 'emotion': 'disgust', 'intensity': 'strong'},
    {'name': '轻微愤怒', 'template': '轻微愤怒', 'category': 'intensity', 'emotion': 'anger', 'intensity': 'weak'},
    {'name': '强烈愤怒', 'template': '强烈愤怒', 'category': 'intensity', 'emotion': 'anger', 'intensity': 'strong'},
    {'name': '微表情惊讶', 'template': '微表情惊讶', 'category': 'intensity', 'emotion': 'surprise', 'intensity': 'micro'},
    {'name': '微表情厌恶', 'template': '微表情厌恶', 'category': 'intensity', 'emotion': 'disgust', 'intensity': 'micro'},
    {'name': '微表情恐惧', 'template': '微表情恐惧', 'category': 'intensity', 'emotion': 'fear', 'intensity': 'micro'},
    {'name': '微表情愤怒', 'template': '微表情愤怒', 'category': 'intensity', 'emotion': 'anger', 'intensity': 'micro'},
    {'name': '微表情 contempt', 'template': '微表情 contempt', 'category': 'intensity', 'emotion': 'contempt', 'intensity': 'micro'},  # 微表情轻蔑

    # Complex - 复合情感
    {'name': '惊讶后微笑', 'template': '先惊讶后微笑', 'category': 'complex', 'emotion': 'surprise_happiness', 'intensity': 'medium'},
    {'name': '恐惧后厌恶', 'template': '先恐惧后厌恶', 'category': 'complex', 'emotion': 'fear_disgust', 'intensity': 'medium'},
    {'name': '愤怒后悲伤', 'template': '先愤怒后悲伤', 'category': 'complex', 'emotion': 'anger_sadness', 'intensity': 'medium'},
    {'name': '惊讶后 contempt', 'template': '先惊讶后 contempt', 'category': 'complex', 'emotion': 'surprise_contempt', 'intensity': 'medium'},  # 惊讶后轻蔑
    {'name': '压抑的微笑', 'template': '压抑的微笑', 'category': 'complex', 'emotion': 'suppressed_happiness', 'intensity': 'medium'},
    {'name': '失望的表情', 'template': '失望的表情', 'category': 'complex', 'emotion': 'disappointment', 'intensity': 'medium'},
]

# =============================================================================
# FeedbackDatabase Class
# =============================================================================

class FeedbackDatabase:
    """SQLite数据库管理类"""

    def __init__(self, db_path: str = None):
        """
        初始化数据库

        Args:
            db_path: 数据库文件路径，默认为 ./feedback_data/feedback.db
        """
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', 'feedback_data', 'feedback.db')

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # 支持字典式访问

        # 初始化表结构
        self._init_schema()
        self._init_default_templates()

        print(f"[FeedbackDatabase] Initialized at {self.db_path}")

    def _init_schema(self):
        """初始化数据库表结构"""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def _init_default_templates(self):
        """初始化默认提示词模板"""
        # 检查是否已有模板
        count = self.conn.execute("SELECT COUNT(*) FROM prompt_templates WHERE is_custom = 0").fetchone()[0]

        if count == 0:
            for template in DEFAULT_PROMPT_TEMPLATES:
                self.conn.execute(
                    """INSERT INTO prompt_templates
                    (name, template, category, emotion, intensity, is_custom, created_at)
                    VALUES (?, ?, ?, ?, ?, 0, ?)""",
                    (template['name'], template['template'], template['category'],
                     template['emotion'], template['intensity'], datetime.now().isoformat())
                )
            self.conn.commit()
            print(f"[FeedbackDatabase] Inserted {len(DEFAULT_PROMPT_TEMPLATES)} default templates")

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def save_feedback(self, feedback: Dict) -> int:
        """
        保存反馈数据

        Args:
            feedback: 反馈字典，包含评分等信息

        Returns:
            feedback_id: 新插入的记录ID
        """
        cursor = self.conn.execute(
            """INSERT INTO feedbacks
            (timestamp, user_id, image_path, prompt, video_path,
             naturalness, smoothness, prompt_match, overall,
             au_accuracy, micro_quality, expert_comments, user_comments,
             emotion, intensity, active_au,
             is_comparison, comparison_winner, quality_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                feedback.get('timestamp', datetime.now().isoformat()),
                feedback.get('user_id', 'anonymous'),
                feedback.get('image_path'),
                feedback.get('prompt', ''),
                feedback.get('video_path'),
                feedback.get('naturalness', 3.0),
                feedback.get('smoothness', 3.0),
                feedback.get('prompt_match', 3.0),
                feedback.get('overall', 3.0),
                feedback.get('au_accuracy'),
                feedback.get('micro_quality'),
                feedback.get('expert_comments'),
                feedback.get('user_comments'),
                feedback.get('emotion'),
                feedback.get('intensity'),
                feedback.get('active_au'),
                feedback.get('is_comparison', 0),
                feedback.get('comparison_winner'),
                self._compute_quality_flag(feedback)
            )
        )

        feedback_id = cursor.lastrowid
        self.conn.commit()

        # 更新提示词使用计数
        if feedback.get('prompt'):
            self.conn.execute(
                "UPDATE prompt_templates SET usage_count = usage_count + 1 WHERE template = ?",
                (feedback['prompt'],)
            )
            self.conn.commit()

        print(f"[FeedbackDatabase] Saved feedback #{feedback_id}")
        return feedback_id

    def _compute_quality_flag(self, feedback: Dict) -> str:
        """计算质量标记"""
        overall = feedback.get('overall', 3.0)
        if overall >= 4.0:
            return 'high'
        elif overall <= 2.0:
            return 'low'
        return 'normal'

    def get_feedback(self, feedback_id: int) -> Optional[Dict]:
        """获取单个反馈"""
        row = self.conn.execute(
            "SELECT * FROM feedbacks WHERE id = ? AND is_deleted = 0",
            (feedback_id,)
        ).fetchone()

        if row:
            return dict(row)
        return None

    def get_all_feedbacks(self, limit: int = 100, offset: int = 0,
                          quality_filter: str = None,
                          min_overall: float = None) -> List[Dict]:
        """
        获取所有反馈列表

        Args:
            limit: 返回数量限制
            offset: 偏移量
            quality_filter: 质量筛选 ('low', 'normal', 'high')
            min_overall: 最小总体评分

        Returns:
            反馈列表
        """
        query = "SELECT * FROM feedbacks WHERE is_deleted = 0"
        params = []

        if quality_filter:
            query += " AND quality_flag = ?"
            params.append(quality_filter)

        if min_overall:
            query += " AND overall >= ?"
            params.append(min_overall)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_feedback(self, feedback_id: int, updates: Dict) -> bool:
        """
        更新反馈数据

        Args:
            feedback_id: 反馈ID
            updates: 更新字段字典

        Returns:
            是否成功
        """
        # 构建动态更新SQL
        allowed_fields = ['naturalness', 'smoothness', 'prompt_match', 'overall',
                          'au_accuracy', 'micro_quality', 'expert_comments', 'user_comments',
                          'quality_flag']

        update_fields = []
        params = []

        for field in allowed_fields:
            if field in updates:
                update_fields.append(f"{field} = ?")
                params.append(updates[field])

        if not update_fields:
            return False

        # 重新计算质量标记
        if 'overall' in updates:
            update_fields.append("quality_flag = ?")
            params.append(self._compute_quality_flag(updates))

        params.append(feedback_id)

        query = f"UPDATE feedbacks SET {', '.join(update_fields)} WHERE id = ?"
        self.conn.execute(query, params)
        self.conn.commit()

        print(f"[FeedbackDatabase] Updated feedback #{feedback_id}")
        return True

    def delete_feedback(self, feedback_id: int, soft_delete: bool = True) -> bool:
        """
        删除反馈

        Args:
            feedback_id: 反馈ID
            soft_delete: 是否软删除（标记删除而非实际删除）

        Returns:
            是否成功
        """
        if soft_delete:
            self.conn.execute(
                "UPDATE feedbacks SET is_deleted = 1 WHERE id = ?",
                (feedback_id,)
            )
        else:
            self.conn.execute(
                "DELETE FROM feedbacks WHERE id = ?",
                (feedback_id,)
            )

        self.conn.commit()
        print(f"[FeedbackDatabase] Deleted feedback #{feedback_id} (soft={soft_delete})")
        return True

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict:
        """获取反馈统计信息"""
        stats = {}

        # 总数
        stats['total_feedbacks'] = self.conn.execute(
            "SELECT COUNT(*) FROM feedbacks WHERE is_deleted = 0"
        ).fetchone()[0]

        # 平均分数
        if stats['total_feedbacks'] > 0:
            avg_row = self.conn.execute(
                """SELECT AVG(naturalness), AVG(smoothness), AVG(prompt_match), AVG(overall)
                FROM feedbacks WHERE is_deleted = 0"""
            ).fetchone()

            stats['avg_naturalness'] = round(avg_row[0], 2)
            stats['avg_smoothness'] = round(avg_row[1], 2)
            stats['avg_prompt_match'] = round(avg_row[2], 2)
            stats['avg_overall'] = round(avg_row[3], 2)

        # 质量分布
        quality_dist = self.conn.execute(
            """SELECT quality_flag, COUNT(*) FROM feedbacks WHERE is_deleted = 0
            GROUP BY quality_flag"""
        ).fetchall()

        stats['quality_distribution'] = {row[0]: row[1] for row in quality_dist}

        # 情感分布
        emotion_dist = self.conn.execute(
            """SELECT emotion, COUNT(*) FROM feedbacks WHERE is_deleted = 0 AND emotion IS NOT NULL
            GROUP BY emotion"""
        ).fetchall()

        stats['emotion_distribution'] = {row[0]: row[1] for row in emotion_dist}

        # 最近反馈
        recent = self.conn.execute(
            "SELECT COUNT(*) FROM feedbacks WHERE is_deleted = 0 AND timestamp >= datetime('now', '-1 day')"
        ).fetchone()[0]
        stats['recent_24h'] = recent

        return stats

    # =========================================================================
    # Prompt Templates
    # =========================================================================

    def get_prompt_templates(self, category: str = None) -> List[Dict]:
        """获取提示词模板"""
        if category:
            rows = self.conn.execute(
                "SELECT * FROM prompt_templates WHERE category = ? ORDER BY usage_count DESC",
                (category,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM prompt_templates ORDER BY category, usage_count DESC"
            ).fetchall()

        return [dict(row) for row in rows]

    def add_custom_template(self, name: str, template: str,
                            category: str = 'basic', emotion: str = None,
                            intensity: str = None) -> int:
        """添加自定义提示词模板"""
        cursor = self.conn.execute(
            """INSERT INTO prompt_templates
            (name, template, category, emotion, intensity, is_custom, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (name, template, category, emotion, intensity, datetime.now().isoformat())
        )

        template_id = cursor.lastrowid
        self.conn.commit()
        print(f"[FeedbackDatabase] Added custom template #{template_id}: {name}")
        return template_id

    def delete_template(self, template_id: int) -> bool:
        """删除提示词模板（仅允许删除自定义模板）"""
        result = self.conn.execute(
            "DELETE FROM prompt_templates WHERE id = ? AND is_custom = 1",
            (template_id,)
        )
        self.conn.commit()
        return result.rowcount > 0

    # =========================================================================
    # Export Functions
    # =========================================================================

    def export_to_json(self, output_path: str = None,
                       quality_filter: str = None,
                       min_overall: float = None) -> str:
        """
        导出为JSON格式（兼容原有格式）

        Args:
            output_path: 输出文件路径
            quality_filter: 质量筛选
            min_overall: 最小评分筛选

        Returns:
            导出文件路径
        """
        feedbacks = self.get_all_feedbacks(
            limit=10000,
            quality_filter=quality_filter,
            min_overall=min_overall
        )

        if output_path is None:
            output_path = str(self.db_path.parent / 'feedback_export.json')

        # 转换为原有格式
        export_data = []
        for fb in feedbacks:
            export_data.append({
                'id': fb['id'],
                'timestamp': fb['timestamp'],
                'image_path': fb['image_path'],
                'prompt': fb['prompt'],
                'video_path': fb['video_path'],
                'naturalness': fb['naturalness'],
                'smoothness': fb['smoothness'],
                'prompt_match': fb['prompt_match'],
                'overall': fb['overall'],
                'au_accuracy': fb['au_accuracy'],
                'micro_quality': fb['micro_quality'],
                'expert_comments': fb['expert_comments'],
                'user_comments': fb['user_comments'],
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"[FeedbackDatabase] Exported {len(export_data)} records to {output_path}")
        return output_path

    def export_to_csv(self, output_path: str = None,
                      quality_filter: str = None,
                      min_overall: float = None) -> str:
        """
        导出为CSV格式

        Returns:
            导出文件路径
        """
        feedbacks = self.get_all_feedbacks(
            limit=10000,
            quality_filter=quality_filter,
            min_overall=min_overall
        )

        if output_path is None:
            output_path = str(self.db_path.parent / 'feedback_export.csv')

        df = pd.DataFrame(feedbacks)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"[FeedbackDatabase] Exported {len(feedbacks)} records to {output_path}")
        return output_path

    def export_to_excel(self, output_path: str = None,
                        quality_filter: str = None,
                        min_overall: float = None) -> str:
        """
        导出为Excel格式

        Returns:
            导出文件路径
        """
        feedbacks = self.get_all_feedbacks(
            limit=10000,
            quality_filter=quality_filter,
            min_overall=min_overall
        )

        if output_path is None:
            output_path = str(self.db_path.parent / 'feedback_export.xlsx')

        df = pd.DataFrame(feedbacks)
        df.to_excel(output_path, index=False, engine='openpyxl')

        print(f"[FeedbackDatabase] Exported {len(feedbacks)} records to {output_path}")
        return output_path

    # =========================================================================
    # Data Cleaning
    # =========================================================================

    def get_low_quality_feedbacks(self) -> List[Dict]:
        """获取低质量反馈列表（用于数据清洗）"""
        return self.get_all_feedbacks(quality_filter='low', limit=1000)

    def batch_delete_low_quality(self, threshold: float = 2.0) -> int:
        """批量删除低质量反馈"""
        result = self.conn.execute(
            "UPDATE feedbacks SET is_deleted = 1 WHERE overall < ? AND is_deleted = 0",
            (threshold,)
        )
        self.conn.commit()
        deleted_count = result.rowcount
        print(f"[FeedbackDatabase] Batch deleted {deleted_count} low quality feedbacks")
        return deleted_count

    def import_from_json(self, json_path: str) -> int:
        """
        从JSON文件导入数据（兼容原有格式）

        Args:
            json_path: JSON文件路径

        Returns:
            导入记录数
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for item in data:
            self.save_feedback(item)
            count += 1

        print(f"[FeedbackDatabase] Imported {count} records from {json_path}")
        return count

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def create_batch_task(self, task_name: str, total_items: int) -> int:
        """创建批量任务"""
        cursor = self.conn.execute(
            """INSERT INTO batch_tasks (task_name, created_at, status, total_items)
            VALUES (?, ?, 'pending', ?)""",
            (task_name, datetime.now().isoformat(), total_items)
        )
        task_id = cursor.lastrowid
        self.conn.commit()
        return task_id

    def update_batch_progress(self, task_id: int, completed: int, status: str = None):
        """更新批量任务进度"""
        if status:
            self.conn.execute(
                "UPDATE batch_tasks SET completed_items = ?, status = ? WHERE id = ?",
                (completed, status, task_id)
            )
        else:
            self.conn.execute(
                "UPDATE batch_tasks SET completed_items = ? WHERE id = ?",
                (completed, task_id)
            )
        self.conn.commit()

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
        print("[FeedbackDatabase] Connection closed")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_db() -> FeedbackDatabase:
    """获取数据库实例（单例模式）"""
    if '_feedback_db' not in globals():
        globals()['_feedback_db'] = FeedbackDatabase()
    return globals()['_feedback_db']


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # 测试
    db = FeedbackDatabase()

    # 测试保存反馈
    test_feedback = {
        'prompt': '微笑',
        'naturalness': 4.0,
        'smoothness': 3.5,
        'prompt_match': 4.5,
        'overall': 4.0,
        'user_comments': '看起来很自然',
    }

    fb_id = db.save_feedback(test_feedback)
    print(f"Test: Saved feedback #{fb_id}")

    # 测试统计
    stats = db.get_statistics()
    print(f"Test: Statistics = {stats}")

    # 测试导出
    db.export_to_json()
    db.export_to_csv()

    # 测试模板
    templates = db.get_prompt_templates()
    print(f"Test: {len(templates)} templates loaded")

    db.close()