-- Run this once in the Supabase SQL Editor for the application project.

insert into storage.buckets (id, name, public, file_size_limit)
values ('user-datasets', 'user-datasets', false, 52428800)
on conflict (id) do update
set public = false, file_size_limit = 52428800;

create table if not exists public.user_datasets (
    id uuid primary key,
    owner_id uuid not null references auth.users (id) on delete cascade,
    filename text not null,
    original_path text not null,
    raw_path text not null,
    clean_path text,
    sheet_name text,
    n_rows integer not null default 0,
    n_cols integer not null default 0,
    clean_rows integer,
    clean_cols integer,
    clean_log jsonb not null default '[]'::jsonb,
    uploaded_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists user_datasets_owner_uploaded_idx
    on public.user_datasets (owner_id, uploaded_at desc);

alter table public.user_datasets enable row level security;
revoke all on table public.user_datasets from anon, authenticated;
grant select, insert, update, delete on table public.user_datasets to authenticated;

drop policy if exists "Users can read their own datasets" on public.user_datasets;
create policy "Users can read their own datasets"
    on public.user_datasets for select to authenticated
    using ((select auth.uid()) = owner_id);

drop policy if exists "Users can create their own datasets" on public.user_datasets;
create policy "Users can create their own datasets"
    on public.user_datasets for insert to authenticated
    with check ((select auth.uid()) = owner_id);

drop policy if exists "Users can update their own datasets" on public.user_datasets;
create policy "Users can update their own datasets"
    on public.user_datasets for update to authenticated
    using ((select auth.uid()) = owner_id)
    with check ((select auth.uid()) = owner_id);

drop policy if exists "Users can delete their own datasets" on public.user_datasets;
create policy "Users can delete their own datasets"
    on public.user_datasets for delete to authenticated
    using ((select auth.uid()) = owner_id);

drop policy if exists "Users can read their own dataset objects" on storage.objects;
create policy "Users can read their own dataset objects"
    on storage.objects for select to authenticated
    using (
        bucket_id = 'user-datasets'
        and (storage.foldername(name))[1] = (select auth.uid()::text)
    );

drop policy if exists "Users can upload their own dataset objects" on storage.objects;
create policy "Users can upload their own dataset objects"
    on storage.objects for insert to authenticated
    with check (
        bucket_id = 'user-datasets'
        and (storage.foldername(name))[1] = (select auth.uid()::text)
    );

drop policy if exists "Users can delete their own dataset objects" on storage.objects;
create policy "Users can delete their own dataset objects"
    on storage.objects for delete to authenticated
    using (
        bucket_id = 'user-datasets'
        and (storage.foldername(name))[1] = (select auth.uid()::text)
    );
