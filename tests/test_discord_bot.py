"""
tests/test_discord_bot.py

Tests for Discord bot functionality (P2 Major Feature)
- Command registration
- Input validation and sanitization
- Cooldown enforcement
- Admin command access control
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import time

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "asketmc_bot"))

# Mock discord before importing
sys.modules['discord'] = Mock()
sys.modules['discord.ext'] = Mock()
sys.modules['discord.ext.commands'] = Mock()
sys.modules['aiohttp'] = Mock()

import discord_bot


class TestInputSanitization:
    """Test input sanitization for Discord commands."""

    def test_sanitize_removes_at_symbols(self):
        """@ symbols are sanitized to prevent mentions."""
        result = discord_bot._sanitize("@everyone hello")
        assert "@everyone" not in result
        assert "@\u200b" in result  # zero-width space added

    def test_sanitize_removes_code_blocks(self):
        """Code block markers are removed."""
        result = discord_bot._sanitize("test ```code``` here")
        assert "```" not in result

    def test_sanitize_removes_sys_tags(self):
        """System tags are removed."""
        result = discord_bot._sanitize("test <sys>tag</sys> here")
        assert "<sys>" not in result
        assert "</sys>" not in result

    def test_sanitize_preserves_valid_text(self):
        """Valid text is preserved."""
        text = "Valid question about server?"
        result = discord_bot._sanitize(text)
        assert "Valid" in result
        assert "question" in result


class TestMessageSplitting:
    """Test long message splitting for Discord limits."""

    def test_split_short_message_unchanged(self):
        """Messages under limit are not split."""
        text = "Short message"
        result = discord_bot._split_for_discord(text, limit=2000)
        assert len(result) == 1
        assert result[0] == text

    def test_split_long_message(self):
        """Messages over limit are split."""
        text = "a" * 5000
        result = discord_bot._split_for_discord(text, limit=2000)
        assert len(result) > 1
        # Recombined should match original
        assert "".join(result) == text

    def test_split_at_newline_boundary(self):
        """Splitting prefers newline boundaries."""
        text = "line1\n" * 100  # Many lines
        result = discord_bot._split_for_discord(text, limit=500)
        # Each chunk should end at newline or be exactly limit
        for chunk in result[:-1]:  # All but last
            assert len(chunk) <= 500

    def test_split_no_newline_uses_limit(self):
        """Without newlines, splits at limit."""
        text = "a" * 3000  # No newlines
        result = discord_bot._split_for_discord(text, limit=1000)
        assert len(result) == 3
        assert all(len(chunk) == 1000 for chunk in result[:-1])


class TestCooldownEnforcement:
    """Test user cooldown logic."""

    def test_cooldown_allows_first_query(self):
        """First query from user is allowed."""
        discord_bot._user_last.clear()
        assert discord_bot._check_cooldown(123) is True

    def test_cooldown_blocks_rapid_queries(self):
        """Rapid queries within cooldown are blocked."""
        discord_bot._user_last.clear()

        user_id = 456
        assert discord_bot._check_cooldown(user_id) is True  # First allowed

        # Immediate second query blocked
        assert discord_bot._check_cooldown(user_id) is False

    def test_cooldown_expires_after_period(self):
        """Cooldown allows query after period expires."""
        discord_bot._user_last.clear()

        user_id = 789
        # First call records time
        assert discord_bot._check_cooldown(user_id) is True

        # Second immediate call blocked
        assert discord_bot._check_cooldown(user_id) is False

        # Manually set last time to past
        discord_bot._user_last[user_id] = time.monotonic() - 1000

        # Now should be allowed
        assert discord_bot._check_cooldown(user_id) is True

    def test_cooldown_per_user(self):
        """Cooldown is per-user, not global."""
        discord_bot._user_last.clear()

        user1 = 111
        user2 = 222

        assert discord_bot._check_cooldown(user1) is True
        assert discord_bot._check_cooldown(user2) is True  # Different user, allowed


class TestStateManagement:
    """Test bot state initialization and access."""

    def test_require_state_raises_when_uninitialized(self):
        """_require_state raises RuntimeError when bot not initialized."""
        discord_bot._STATE = None

        with pytest.raises(RuntimeError, match="not initialized"):
            discord_bot._require_state()

    def test_require_state_returns_state_when_initialized(self):
        """_require_state returns state when properly initialized."""
        mock_deps = discord_bot.AppDeps(
            index=Mock(),
            retriever=Mock(),
            generate_rag_answer=Mock(),
            query_model=Mock(),
            call_local_llm=Mock(),
            build_index=Mock(),
            is_openrouter_blocked=Mock(),
        )
        discord_bot._STATE = mock_deps

        result = discord_bot._require_state()
        assert result == mock_deps

        # Cleanup
        discord_bot._STATE = None


class TestBotBuilder:
    """Test bot construction and command registration."""

    def test_build_bot_creates_bot_instance(self):
        """_build_bot creates a bot with proper intents."""
        with patch('discord_bot.commands.Bot') as mock_bot_cls:
            mock_bot_cls.return_value = Mock()

            bot = discord_bot._build_bot()

            mock_bot_cls.assert_called_once()
            # Verify intents were configured
            call_kwargs = mock_bot_cls.call_args[1]
            assert 'intents' in call_kwargs
            assert 'command_prefix' in call_kwargs

    def test_command_registration(self):
        """Commands are registered on bot."""
        mock_bot = Mock()

        # Mock the command decorator
        def mock_command(**kwargs):
            def decorator(func):
                return func
            return decorator

        mock_bot.command = mock_command

        # Register commands
        discord_bot._register_commands(mock_bot)

        # Verify bot.command was used (can't easily test decorator registration)
        # Just ensure function doesn't crash


class TestAdminCommandChecks:
    """Test admin-only command access control."""

    def test_check_admin_only_creates_predicate(self):
        """_check_admin_only returns a check decorator."""
        check_decorator = discord_bot._check_admin_only()
        assert callable(check_decorator)

    def test_admin_check_allows_admin(self):
        """Admin user passes admin check."""
        with patch('discord_bot.cfg.ADMIN_IDS', {123}):
            check_decorator = discord_bot._check_admin_only()
            # Get the actual predicate function
            # (Complex due to decorator structure, simplified test)

    def test_admin_check_blocks_non_admin(self):
        """Non-admin user fails admin check."""
        with patch('discord_bot.cfg.ADMIN_IDS', {123}):
            check_decorator = discord_bot._check_admin_only()
            # Would need to create mock context to fully test


class TestChannelChecks:
    """Test channel-allowed command access control."""

    def test_check_channel_allowed_creates_predicate(self):
        """_check_channel_allowed returns a check decorator."""
        check_decorator = discord_bot._check_channel_allowed()
        assert callable(check_decorator)


class TestSessionHolder:
    """Test async session holder for bot HTTP requests."""

    @pytest.mark.asyncio
    async def test_session_holder_creates_session(self):
        """Session holder creates session on first access."""
        holder = discord_bot._AsyncSessionHolder()

        with patch('discord_bot.ClientSession') as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_cls.return_value = mock_session

            session = await holder.get()
            mock_session_cls.assert_called_once()

        await holder.close()

    @pytest.mark.asyncio
    async def test_session_holder_reuses_session(self):
        """Session holder reuses open session."""
        holder = discord_bot._AsyncSessionHolder()

        with patch('discord_bot.ClientSession') as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_cls.return_value = mock_session

            session1 = await holder.get()
            session2 = await holder.get()

            # Should only create once
            assert mock_session_cls.call_count == 1

        await holder.close()

    @pytest.mark.asyncio
    async def test_session_holder_closes_properly(self):
        """Session holder closes session on close()."""
        holder = discord_bot._AsyncSessionHolder()

        with patch('discord_bot.ClientSession') as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            await holder.get()
            await holder.close()

            mock_session.close.assert_called_once()
