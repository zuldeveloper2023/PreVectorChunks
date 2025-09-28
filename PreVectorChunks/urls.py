from django.urls import path
from .api import views

urlpatterns = [
    path('lms/', views.queryUnitContent, name='lms'),
    path('uploadlsmcontent/', views.upload_json_file, name='lms'),

    path('upsertlmscontent/', views.upsertContentToVDB, name='lms'),

    path('upsertlmscontentinchunks/', views.upsertContentToVDB_Chunks, name='lms'),

    path('fetchrecordsbyuniquedoconame/', views.fetch_records_grouped_by_document_name, name='lms'),
    path('updaterecordsbydoconame/', views.update_records_grouped_by_document_name_in_vdb, name='lms'),


]
