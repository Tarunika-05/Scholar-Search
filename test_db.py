import asyncio
import sys
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def main():
    url = 'postgresql+asyncpg://neondb_owner:npg_3ioaDeB5rpAg@ep-cold-union-atqa6k3p-pooler.c-9.us-east-1.aws.neon.tech/neondb?ssl=require'
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text('SELECT 1'))
            print('Success!', result.scalar())
    except Exception as e:
        print('Error:', e)
        sys.exit(1)

asyncio.run(main())
