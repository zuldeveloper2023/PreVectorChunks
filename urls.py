from django.urls import path
from web.prevectorchunks_web.api import views

urlpatterns = [
    path('lms/', views.queryUnitContent, name='lms'),
    path('uploadlsmcontent/', views.chunk_documents_endpoint, name='lms'),
    path('upsertlmscontentinchunks/', views.chunk_documents_and_upsert_into_vdb_endpoint, name='lms'),
    path('fetchrecordsbyuniquedoconame/', views.fetch_records_grouped_by_document_name, name='lms'),
    path('updaterecordsbydoconame/', views.update_records_grouped_by_document_name_in_vdb, name='lms'),
]
