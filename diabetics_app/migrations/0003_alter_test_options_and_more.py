# Generated by Django 4.0.6 on 2022-08-02 21:46

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('diabetics_app', '0002_alter_test_result_alter_test_slug'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='test',
            options={'verbose_name_plural': "Patients' Record"},
        ),
        migrations.RenameField(
            model_name='test',
            old_name='left_eye',
            new_name='generated_data',
        ),
        migrations.RenameField(
            model_name='test',
            old_name='right_eye',
            new_name='uploaded_data',
        ),
    ]
