#! /bin/bash

psql -U "$POSTGRES_USER" -W -d "$POSTGRES_DB" -f grocery.sql
