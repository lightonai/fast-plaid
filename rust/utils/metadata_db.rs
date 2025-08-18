use anyhow::{Context, Result};
use duckdb::{params, Connection};
use serde_json::Value;
use std::collections::HashSet;
use std::path::Path;

/// Manages DuckDB database for storing document metadata and centroid mappings
pub struct MetadataDB {
    conn: Connection,
}

impl MetadataDB {
    /// Creates a new MetadataDB connection to the database file in the index directory
    pub fn new(index_path: &str) -> Result<Self> {
        let db_path = Path::new(index_path).join("metadata.duckdb");
        let conn = Connection::open(db_path)
            .context("Failed to open DuckDB connection")?;
        
        let db = Self { conn };
        db.create_tables()?;
        Ok(db)
    }

    /// Creates the necessary tables for metadata storage
    fn create_tables(&self) -> Result<()> {
        // Documents table to store metadata
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                passage_id BIGINT PRIMARY KEY,
                metadata JSON
            )",
            [],
        ).context("Failed to create documents table")?;

        // Centroid to document mapping for efficient filtering
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS centroid_documents (
                centroid_id BIGINT,
                passage_id BIGINT,
                PRIMARY KEY (centroid_id, passage_id)
            )",
            [],
        ).context("Failed to create centroid_documents table")?;

        // Index for faster lookups
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_centroid_documents_centroid 
             ON centroid_documents (centroid_id)",
            [],
        ).context("Failed to create centroid index")?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_centroid_documents_passage 
             ON centroid_documents (passage_id)",
            [],
        ).context("Failed to create passage index")?;

        Ok(())
    }

    /// Stores document metadata and their centroid mappings
    pub fn store_documents(&self, documents_metadata: &[Value], centroid_mappings: &[(i64, i64)]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()
            .context("Failed to start transaction")?;

        // Clear existing data (for create, not update)
        tx.execute("DELETE FROM documents", [])
            .context("Failed to clear documents table")?;
        tx.execute("DELETE FROM centroid_documents", [])
            .context("Failed to clear centroid_documents table")?;

        // Insert document metadata
        for (passage_id, metadata) in documents_metadata.iter().enumerate() {
            tx.execute(
                "INSERT INTO documents (passage_id, metadata) VALUES (?, ?)",
                params![passage_id as i64, metadata.to_string()],
            ).with_context(|| format!("Failed to insert metadata for passage {}", passage_id))?;
        }

        // Insert centroid mappings
        for &(centroid_id, passage_id) in centroid_mappings {
            tx.execute(
                "INSERT OR IGNORE INTO centroid_documents (centroid_id, passage_id) VALUES (?, ?)",
                params![centroid_id, passage_id],
            ).with_context(|| format!("Failed to insert centroid mapping {}:{}", centroid_id, passage_id))?;
        }

        tx.commit().context("Failed to commit transaction")?;
        Ok(())
    }

    /// Updates existing documents with new metadata and centroid mappings (for index updates)
    pub fn update_documents(&self, documents_metadata: &[Value], centroid_mappings: &[(i64, i64)], passage_id_offset: i64) -> Result<()> {
        let tx = self.conn.unchecked_transaction()
            .context("Failed to start transaction")?;

        // Insert new document metadata
        for (local_passage_id, metadata) in documents_metadata.iter().enumerate() {
            let global_passage_id = passage_id_offset + local_passage_id as i64;
            tx.execute(
                "INSERT INTO documents (passage_id, metadata) VALUES (?, ?)",
                params![global_passage_id, metadata.to_string()],
            ).with_context(|| format!("Failed to insert metadata for passage {}", global_passage_id))?;
        }

        // Insert new centroid mappings
        for &(centroid_id, passage_id) in centroid_mappings {
            tx.execute(
                "INSERT OR IGNORE INTO centroid_documents (centroid_id, passage_id) VALUES (?, ?)",
                params![centroid_id, passage_id],
            ).with_context(|| format!("Failed to insert centroid mapping {}:{}", centroid_id, passage_id))?;
        }

        tx.commit().context("Failed to commit transaction")?;
        Ok(())
    }

    /// Finds centroids that contain documents matching the filter query
    pub fn get_centroids_for_filter(&self, filter_query: &str) -> Result<HashSet<i64>> {
        let query = format!(
            "SELECT DISTINCT cd.centroid_id 
             FROM centroid_documents cd 
             JOIN documents d ON cd.passage_id = d.passage_id 
             WHERE {}",
            filter_query
        );

        let mut stmt = self.conn.prepare(&query)
            .with_context(|| format!("Failed to prepare centroid filter query: {}", query))?;

        let rows = stmt.query_map([], |row| {
            Ok(row.get::<_, i64>(0)?)
        }).context("Failed to execute centroid filter query")?;

        let mut centroids = HashSet::new();
        for row in rows {
            centroids.insert(row.context("Failed to read centroid ID from result")?);
        }

        Ok(centroids)
    }

    /// Filters passage IDs based on metadata query
    pub fn filter_passages(&self, passage_ids: &[i64], filter_query: &str) -> Result<Vec<i64>> {
        if passage_ids.is_empty() {
            return Ok(vec![]);
        }

        let placeholders = (0..passage_ids.len())
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");

        let query = format!(
            "SELECT d.passage_id 
             FROM documents d 
             WHERE d.passage_id IN ({}) AND {}",
            placeholders, filter_query
        );

        let mut stmt = self.conn.prepare(&query)
            .with_context(|| format!("Failed to prepare passage filter query: {}", query))?;

        let params: Vec<&dyn duckdb::ToSql> = passage_ids.iter()
            .map(|id| id as &dyn duckdb::ToSql)
            .collect();

        let rows = stmt.query_map(params.as_slice(), |row| {
            Ok(row.get::<_, i64>(0)?)
        }).context("Failed to execute passage filter query")?;

        let mut filtered_passages = Vec::new();
        for row in rows {
            filtered_passages.push(row.context("Failed to read passage ID from result")?);
        }

        Ok(filtered_passages)
    }

    /// Gets the total number of documents in the database
    pub fn get_document_count(&self) -> Result<i64> {
        let mut stmt = self.conn.prepare("SELECT COUNT(*) FROM documents")
            .context("Failed to prepare count query")?;

        let count = stmt.query_row([], |row| {
            Ok(row.get::<_, i64>(0)?)
        }).context("Failed to get document count")?;

        Ok(count)
    }

    /// Checks if the database exists and has data
    pub fn exists_and_has_data(index_path: &str) -> bool {
        let db_path = Path::new(index_path).join("metadata.duckdb");
        if !db_path.exists() {
            return false;
        }

        // Try to connect and check if tables exist with data
        if let Ok(conn) = Connection::open(&db_path) {
            if let Ok(mut stmt) = conn.prepare("SELECT COUNT(*) FROM documents") {
                if let Ok(count) = stmt.query_row([], |row| Ok(row.get::<_, i64>(0)?)) {
                    return count > 0;
                }
            }
        }

        false
    }
}