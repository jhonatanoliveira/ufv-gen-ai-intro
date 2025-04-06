"""Database models for news data storage."""

from peewee import CharField, DateTimeField, Model, SqliteDatabase, TextField

# SQLite database instance
db_client = SqliteDatabase("news.db")


class BaseModel(Model):
    """Base model class that sets the database connection."""

    class Meta:
        """Meta configuration for database connection."""

        database = db_client


class NewsModel(BaseModel):
    """Model representing a news article."""

    code = CharField(primary_key=True)  # Unique identifier for the news article
    category = CharField()  # News category
    title = CharField()  # Article title
    date = DateTimeField()  # Publication date
    content = TextField()  # Full article content
    location = CharField()  # Geographic location of the news

    class Meta:
        """Meta configuration for table name."""

        table_name = "scraped_news"


__all__ = ["NewsModel", "db_client"]
